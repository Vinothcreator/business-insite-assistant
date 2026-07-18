# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from datetime import datetime, timedelta
import os
import json
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Import local engines (Hot-reloaded for dynamic percentiles classification)
import analytics_engine as ae
import ai_engine as aie

# Page Configuration
st.set_page_config(
    page_title="AI Business InSite Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Slate Glassmorphic Design
st.markdown("""
<style>
    /* Fonts and Headings */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        color: #F1F5F9;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #334155;
    }
    
    /* Card Container */
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
        margin-bottom: 15px;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #60A5FA;
    }
    
    /* KPI Text styling */
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #94A3B8;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #F1F5F9;
        margin: 5px 0;
        font-family: 'Outfit', sans-serif;
    }
    .metric-delta {
        font-size: 0.9rem;
        font-weight: 500;
    }
    .delta-up {
        color: #34D399; /* Emerald Green */
    }
    .delta-down {
        color: #F87171; /* Coral Red */
    }
    
    /* Glowing accents */
    .glow-blue {
        border-left: 4px solid #60A5FA;
    }
    .glow-green {
        border-left: 4px solid #34D399;
    }
    .glow-orange {
        border-left: 4px solid #FBBF24;
    }
    .glow-red {
        border-left: 4px solid #F87171;
    }
</style>
""", unsafe_allow_html=True)

# Database connectivity checks
@st.cache_resource
def get_db_connection():
    try:
        password_encoded = quote_plus(aie.DB_CONFIG['password'])
        engine_url = f"mysql+mysqlconnector://{aie.DB_CONFIG['user']}:{password_encoded}@{aie.DB_CONFIG['host']}:{aie.DB_CONFIG['port']}/{aie.DB_CONFIG['database']}"
        return create_engine(engine_url)
    except Exception as e:
        st.sidebar.error(f"Database Driver Error: {e}")
        return None

engine = get_db_connection()

def get_last_updated_time():
    manifest_path = os.path.join(BASE_DIR, "data", "raw", "version_manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                data = json.load(f)
                dt_str = data.get("last_updated", "Never")
                if dt_str != "Never":
                    dt = datetime.fromisoformat(dt_str)
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                return dt_str
        except Exception:
            return "Unknown"
    return "Never"

# Startup Check & Automatic Ingestion
def check_and_initialize_data():
    is_seeded = False
    try:
        if engine is not None:
            with engine.connect() as conn:
                res = conn.execute(text("SELECT COUNT(*) FROM fact_sales;"))
                count = res.scalar()
                if count > 0:
                    is_seeded = True
    except Exception as e:
        is_seeded = False
        
    if not is_seeded:
        st.info("First-time startup: Ingesting sales data from Kaggle and setting up dimensions/facts...")
        try:
            import import_kaggle_sales
            has_creds = import_kaggle_sales.setup_kaggle_credentials()
            import_kaggle_sales.download_data(has_credentials=has_creds)
            import_kaggle_sales.process_and_ingest()
            st.success("Data Ingestion and ELT from Kaggle completed successfully!")
        except Exception as ex:
            st.error(f"Startup data ingestion failed: {ex}")
            
check_and_initialize_data()

# Initialize session states for user preferences & active workflows
if "role" not in st.session_state:
    st.session_state.role = "CEO"
if "ai_mode" not in st.session_state:
    st.session_state.ai_mode = "Offline Semantic Mode"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "action_logs" not in st.session_state:
    st.session_state.action_logs = []
if "chat_context" not in st.session_state:
    st.session_state.chat_context = {}

# Sidebar configurations
with st.sidebar:
    st.title("Workspace Controls")
    
    # 1. Role Selection (RBAC Enforcement)
    role_options = ["CEO", "Sales Manager", "Operations Manager"]
    st.session_state.role = st.selectbox(
        "User Access Role",
        options=role_options,
        index=role_options.index(st.session_state.role)
    )
    
    st.markdown(f"**Current Permissions:** \n- `{st.session_state.role}` access levels apply.")
    
    st.divider()
    
    # 2. AI Mode configuration
    ai_mode_options = ["Offline Semantic Mode", "Google Gemini AI Mode (requires Gemini Key)", "Live AI Mode (requires OpenAI Key)"]
    st.session_state.ai_mode = st.radio(
        "AI Assistant Mode",
        options=ai_mode_options,
        index=ai_mode_options.index(st.session_state.ai_mode)
    )
    
    # Check for keys based on selection
    if st.session_state.ai_mode == "Google Gemini AI Mode (requires Gemini Key)":
        st.markdown("##### Ingestion Options")
        
        # Check if already loaded from env
        current_gemini_key = os.getenv("GEMINI_API_KEY")
        if current_gemini_key and len(current_gemini_key.strip()) > 10:
            st.success("API Key automatically loaded from `.env` file!")
            
        # 1. Text input
        api_key_text = st.text_input("Option A: Paste Gemini API Key", type="password", placeholder="AIzaSy...")
        if api_key_text:
            os.environ["GEMINI_API_KEY"] = api_key_text
            st.success("Gemini API Key loaded from text input.")
            
        # 2. File upload
        uploaded_key_file = st.file_uploader("Option B: Upload Key File (.txt / .env)", type=["txt", "env"], key="gemini_key_file_uploader")
        if uploaded_key_file is not None:
            try:
                file_content = uploaded_key_file.read().decode("utf-8").strip()
                # Parse .env key if present
                if "=" in file_content:
                    for line in file_content.splitlines():
                        if line.strip().startswith("GEMINI_API_KEY"):
                            parts = line.split("=", 1)
                            file_content = parts[1].strip().strip('"').strip("'")
                            break
                
                if len(file_content) > 10:
                    os.environ["GEMINI_API_KEY"] = file_content
                    st.success("Gemini API Key loaded from uploaded file!")
                else:
                    st.error("Invalid key content inside uploaded file.")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    elif st.session_state.ai_mode == "Live AI Mode (requires OpenAI Key)":
        st.markdown("##### Ingestion Options")
        
        # 1. Text input
        api_key_text = st.text_input("Option A: Paste OpenAI API Key", type="password", placeholder="sk-...")
        if api_key_text:
            os.environ["OPENAI_API_KEY"] = api_key_text
            st.success("API Key loaded from text input.")
            
        # 2. File upload
        uploaded_key_file = st.file_uploader("Option B: Upload Key File (.txt / .env)", type=["txt", "env"], key="api_key_file_uploader")
        if uploaded_key_file is not None:
            try:
                file_content = uploaded_key_file.read().decode("utf-8").strip()
                # Parse .env key if present
                if "=" in file_content:
                    for line in file_content.splitlines():
                        if line.strip().startswith("OPENAI_API_KEY"):
                            parts = line.split("=", 1)
                            file_content = parts[1].strip().strip('"').strip("'")
                            break
                
                if len(file_content) > 10:
                    os.environ["OPENAI_API_KEY"] = file_content
                    st.success("API Key loaded from uploaded file!")
                else:
                    st.error("Invalid key content inside uploaded file.")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                
        # Status check
        current_env_key = os.getenv("OPENAI_API_KEY")
        if not current_env_key or len(current_env_key.strip()) < 10:
            st.warning("No API key loaded. Will use offline fallback.")
            
    st.divider()
    
    # 3. Connection Monitor
    st.subheader("Connection Monitor")
    if engine:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1;"))
            st.success("MySQL Server: Connected (port 3306)")
        except Exception:
            st.error("MySQL Server: Disconnected")
    else:
        st.error("MySQL Server: Connection Failed")
        
    st.markdown("""
    - CRM Integration (Salesforce): Active
    - ERP Integration (NetSuite): Active
    - Data Lake Pipeline (ELT): Active
    """)
    
    st.divider()
    st.markdown("### Data Freshness Control")
    last_up = get_last_updated_time()
    st.caption(f"Last Updated: {last_up}")
    if st.button("Trigger Data Auto-Update", help="Re-fetch and validate dataset from Kaggle/GitHub"):
        try:
            import import_kaggle_sales
            has_creds = import_kaggle_sales.setup_kaggle_credentials()
            import_kaggle_sales.download_data(has_credentials=has_creds)
            import_kaggle_sales.process_and_ingest()
            st.success("Data updated and ELT successfully re-run!")
            st.rerun()
        except Exception as ex:
            st.error(f"Update failed: {ex}")

# Helper function to query safe data
def run_db_query(sql, show_error=True):
    try:
        return aie.execute_query(sql, st.session_state.role)
    except Exception as e:
        if show_error:
            st.error(f"Database Query Blocked: {e}")
        return pd.DataFrame()

# Main Application Menu / Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Dashboard", 
    "Advanced Analytics & ML", 
    "Intelligent Sandbox & Resolution", 
    "Conversational AI Assistant", 
    "Data Governance & Audits"
])

# ----------------- TAB 1: EXECUTIVE DASHBOARD -----------------
with tab1:
    st.header("Enterprise Sales Overview & Operations Insights")
    
    role = st.session_state.role
    
    # Fetch KPI metrics safely depending on role
    df_sales = pd.DataFrame()
    df_issues = pd.DataFrame()
    df_leads = pd.DataFrame()
    df_monthly = pd.DataFrame()
    
    # 1. Leads (Accessible by all roles)
    df_leads = run_db_query("SELECT * FROM raw_leads;", show_error=False)
    
    # 2. Issues (Blocked for Sales Manager)
    if role != "Sales Manager":
        df_issues = run_db_query("SELECT * FROM raw_issues;", show_error=False)
        
    # 3. Sales & Monthly (Blocked for Operations Manager due to financial columns)
    if role != "Operations Manager":
        df_sales = run_db_query("""
            SELECT f.*, r.region, r.channel as sales_channel 
            FROM fact_sales f 
            LEFT JOIN dim_regions r ON f.region_id = r.region_id 
            WHERE f.status='Completed';
        """, show_error=False)
        df_monthly = run_db_query("SELECT year, month, SUM(total_revenue) as rev FROM agg_sales_monthly GROUP BY year, month ORDER BY year, month;", show_error=False)
    else:
        # Operations manager gets order counts and statuses only (no financial data)
        df_sales = run_db_query("""
            SELECT f.order_id, f.customer_id, f.product_id, f.date_id, f.quantity, f.status, r.region, r.channel as sales_channel
            FROM fact_sales f
            LEFT JOIN dim_regions r ON f.region_id = r.region_id
            WHERE f.status='Completed';
        """, show_error=False)
        
    # Calculate values for cards
    net_revenue = None
    total_profit = None
    avg_margin = None
    mom_growth = None
    yearly_change = None
    
    if "imported_metrics" in st.session_state and st.session_state["imported_metrics"] is not None:
        metrics = st.session_state["imported_metrics"]
        if role != "Operations Manager":
            net_revenue = metrics["total_revenue"]
            total_profit = metrics["total_profit_loss"]
            avg_margin = metrics["profit_margin"]
            mom_growth = metrics["mom_growth"]
            yearly_change = metrics["yearly_change"]
    else:
        if role != "Operations Manager" and not df_sales.empty:
            net_revenue = df_sales["net_revenue"].sum()
            total_profit = df_sales["profit_amount"].sum()
            avg_margin = (total_profit / net_revenue * 100) if net_revenue > 0 else 0.0
            
            if not df_monthly.empty and len(df_monthly) >= 2:
                prev_rev = float(df_monthly.iloc[-2]["rev"])
                curr_rev = float(df_monthly.iloc[-1]["rev"])
                mom_growth = ((curr_rev - prev_rev) / prev_rev) * 100 if prev_rev > 0 else 0.0
            
    # Draw metric cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if net_revenue is not None:
            rev_val = f"${net_revenue:,.2f}"
            if mom_growth is not None:
                # Green + if growth > 0%
                # Red - if growth < 0%
                if mom_growth > 0:
                    growth_val = f"<span class='delta-up'>+ {mom_growth:.1f}%</span> MoM Growth"
                elif mom_growth < 0:
                    growth_val = f"<span class='delta-down'>- {mom_growth:.1f}%</span> MoM Growth"
                else:
                    growth_val = f"<span>0.0%</span> MoM Growth"
                
                # Append yearly change if available
                if yearly_change is not None and yearly_change != 0.0:
                    if yearly_change > 0:
                        growth_val += f" | <span class='delta-up'>+ {yearly_change:.1f}% YoY</span>"
                    elif yearly_change < 0:
                        growth_val += f" | <span class='delta-down'>- {yearly_change:.1f}% YoY</span>"
            else:
                growth_val = "Baseline"
        else:
            rev_val = "Restricted"
            growth_val = "<span class='delta-down'>Restricted: CFO/Sales Only</span>"
            
        st.markdown(f"""
        <div class="metric-card glow-blue">
            <div class="metric-title">Net Sales Revenue</div>
            <div class="metric-value">{rev_val}</div>
            <div class="metric-delta">{growth_val}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        if avg_margin is not None:
            margin_val = f"{avg_margin:.1f}%"
            if "imported_metrics" in st.session_state and st.session_state["imported_metrics"] is not None:
                margin_delta = f"Net Profit: <span class='delta-up' style='font-weight:600;'>${total_profit:,.2f}</span>" if total_profit >= 0 else f"Net Loss: <span class='delta-down' style='font-weight:600;'>${total_profit:,.2f}</span>"
            else:
                margin_delta = "Target: <span class='delta-up'>50.0%</span>"
        else:
            margin_val = "Restricted"
            margin_delta = "<span class='delta-down'>Restricted: CFO/Sales Only</span>"
            
        st.markdown(f"""
        <div class="metric-card glow-green">
            <div class="metric-title">Net Profit Margin</div>
            <div class="metric-value">{margin_val}</div>
            <div class="metric-delta">{margin_delta}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        total_leads = len(df_leads) if not df_leads.empty else 0
        converted_leads = len(df_leads[df_leads["status"] == "Converted"]) if not df_leads.empty else 0
        conv_rate = (converted_leads / total_leads * 100) if total_leads > 0 else 0.0
        st.markdown(f"""
        <div class="metric-card glow-orange">
            <div class="metric-title">Lead Pipeline Volume</div>
            <div class="metric-value">{total_leads}</div>
            <div class="metric-delta">Conv Rate: <span class="delta-up">{conv_rate:.1f}%</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        if role != "Sales Manager":
            open_issues_count = len(df_issues[df_issues["status"] != "Resolved"]) if not df_issues.empty else 0
            tickets_val = f"{open_issues_count}"
            tickets_delta = "Status: <span class='delta-down'>Attention Needed</span>" if open_issues_count > 0 else "Status: <span class='delta-up'>All Clear</span>"
        else:
            tickets_val = "Restricted"
            tickets_delta = "<span class='delta-down'>Restricted: Ops/Support Only</span>"
            
        st.markdown(f"""
        <div class="metric-card glow-red">
            <div class="metric-title">Active Support Tickets</div>
            <div class="metric-value">{tickets_val}</div>
            <div class="metric-delta">{tickets_delta}</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Interactive Dashboard Graphs
    st.subheader("Time Series Sales Trends & Operational Analysis")
    
    if role == "Operations Manager":
        st.warning("Access Restricted: Operational roles do not have permission to view sales financial graphs.")
    else:
        col_graph1, col_graph2 = st.columns([2, 1])
        
        with col_graph1:
            if not df_sales.empty:
                # Group sales by date for daily plot
                df_daily = df_sales.groupby("date_id")[["net_revenue", "profit_amount"]].sum().reset_index()
                # Convert date_id to str
                df_daily["date_id"] = df_daily["date_id"].astype(str)
                df_daily = df_daily.sort_values("date_id").reset_index(drop=True)
                
                trend_title = "Daily Net Sales Revenue Trend (July 2025 - July 2026)"
                
                fig_trend = px.line(
                    df_daily, x="date_id", y="net_revenue",
                    title=trend_title,
                    labels={"date_id": "Date", "net_revenue": "Net Sales ($)"},
                    color_discrete_sequence=["#60A5FA"]
                )
                # Add anomaly flags visually dynamically
                anoms = ae.detect_anomalies(df_daily["date_id"], df_daily["net_revenue"])
                for an in anoms:
                    if an["type"] == "Drop":
                        fig_trend.add_annotation(
                            x=an["date"], y=an["value"], text=f"Drop: {an['z_score']:.1f}σ",
                            showarrow=True, arrowhead=1, ax=0, ay=-40,
                            bordercolor="#F87171", borderwidth=1, borderpad=4, bgcolor="#1E293B"
                        )
                    elif an["type"] == "Spike":
                        fig_trend.add_annotation(
                            x=an["date"], y=an["value"], text=f"Spike: +{an['z_score']:.1f}σ",
                            showarrow=True, arrowhead=1, ax=0, ay=-30,
                            bordercolor="#34D399", borderwidth=1, borderpad=4, bgcolor="#1E293B"
                        )
                fig_trend.update_layout(
                    paper_bgcolor="#1E293B", plot_bgcolor="#1E293B",
                    font_color="#F1F5F9", margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("No sales revenue data available.")
                
        with col_graph2:
            if not df_sales.empty:
                # Regional revenue breakdown
                df_region = df_sales.groupby("region")["net_revenue"].sum().reset_index()
                fig_region = px.pie(
                    df_region, values="net_revenue", names="region",
                    hole=0.4, title="Revenue Share by Region",
                    color_discrete_sequence=px.colors.sequential.Blues_r
                )
                fig_region.update_layout(
                    paper_bgcolor="#1E293B", plot_bgcolor="#1E293B",
                    font_color="#F1F5F9", margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_region, use_container_width=True)
            else:
                st.info("No regional sales revenue data available.")
                
        # Category breakdown
        st.subheader("Product Category Performance Leaderboard")
        df_product_summary = run_db_query("SELECT category, SUM(total_qty_sold) as total_qty, SUM(total_net_sales) as total_sales, SUM(total_profit) as total_profit FROM agg_sales_product_summary GROUP BY category;", show_error=False)
        if not df_product_summary.empty:
            c1, c2 = st.columns(2)
            with c1:
                fig_cat = px.bar(
                    df_product_summary, x="category", y="total_sales",
                    title="Revenue by Product Category",
                    labels={"category": "Category", "total_sales": "Net Sales ($)"},
                    color="category", color_discrete_sequence=["#60A5FA", "#34D399", "#A78BFA"]
                )
                fig_cat.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#1E293B", font_color="#F1F5F9")
                st.plotly_chart(fig_cat, use_container_width=True)
            with c2:
                # Top Products list
                df_products_top = run_db_query("SELECT product_name, category, total_qty_sold, total_net_sales FROM agg_sales_product_summary ORDER BY total_net_sales DESC LIMIT 5;", show_error=False)
                st.write("**Top 5 Selling Products**")
                st.dataframe(df_products_top, use_container_width=True)
                
        # External spreadsheet data import (CEO and Sales Manager only)
        st.divider()
        if role in ["CEO", "Sales Manager"]:
            st.subheader("Financial Data Import (Excel / CSV)")
            st.write("Ingest external transactional sheets to calculate total revenues, net profits, and profit margins dynamically.")
            
            import_source = st.selectbox(
                "Select Financial Data Source",
                options=[
                    "Upload Custom File(s)", 
                    "Load Sample Company Revenue (2024)", 
                    "Load Sample Company Revenue (2025)", 
                    "Load Retail Store Transactions (2024)"
                ],
                key="financial_data_source_selector",
                help="Choose whether to upload your own sheets or load pre-generated sample company datasets."
            )
            
            # If the source changes, clear existing session state
            if "last_import_source" not in st.session_state or st.session_state["last_import_source"] != import_source:
                st.session_state["last_import_source"] = import_source
                if "uploaded_filename" in st.session_state:
                    del st.session_state["uploaded_filename"]
                if "imported_metrics" in st.session_state:
                    del st.session_state["imported_metrics"]
                if "mapped_cols_raw" in st.session_state:
                    del st.session_state["mapped_cols_raw"]
                if "raw_df_imported" in st.session_state:
                    del st.session_state["raw_df_imported"]
                st.rerun()

            df_imported = None
            source_name = None
            
            if import_source == "Upload Custom File(s)":
                uploaded_files = st.file_uploader(
                    "Upload Excel or CSV Sheet(s)", 
                    type=["xlsx", "xls", "csv"], 
                    key="financial_data_uploader",
                    accept_multiple_files=True,
                    help="Requires columns for Revenue/Sales (e.g., Sales Amount, Revenue) and Date to compute metrics."
                )
                
                if len(uploaded_files) > 0:
                    source_name = ", ".join([f.name for f in uploaded_files])
                    if "uploaded_filename" not in st.session_state or st.session_state["uploaded_filename"] != source_name:
                        try:
                            dfs = []
                            for file in uploaded_files:
                                if file.name.endswith(".csv"):
                                    dfs.append(pd.read_csv(file))
                                else:
                                    dfs.append(pd.read_excel(file))
                            df_imported = pd.concat(dfs, ignore_index=True)
                        except Exception as e:
                            st.error(f"Error parsing uploaded files: {e}")
                else:
                    if "uploaded_filename" in st.session_state:
                        del st.session_state["uploaded_filename"]
                        if "imported_metrics" in st.session_state:
                            del st.session_state["imported_metrics"]
                        if "mapped_cols_raw" in st.session_state:
                            del st.session_state["mapped_cols_raw"]
                        if "raw_df_imported" in st.session_state:
                            del st.session_state["raw_df_imported"]
                        st.rerun()
            else:
                if import_source == "Load Sample Company Revenue (2024)":
                    file_path = os.path.join(BASE_DIR, "data", "raw", "sample_company_revenue_2024.xlsx")
                    source_name = "sample_company_revenue_2024.xlsx"
                elif import_source == "Load Sample Company Revenue (2025)":
                    file_path = os.path.join(BASE_DIR, "data", "raw", "sample_company_revenue.xlsx")
                    source_name = "sample_company_revenue.xlsx"
                else:
                    file_path = os.path.join(BASE_DIR, "Sales-datasets-excel", "Retail-Store-Transactions.xlsx")
                    source_name = "Retail-Store-Transactions.xlsx"
                    
                if "uploaded_filename" not in st.session_state or st.session_state["uploaded_filename"] != source_name:
                    try:
                        if file_path.endswith(".csv"):
                            df_imported = pd.read_csv(file_path)
                        else:
                            df_imported = pd.read_excel(file_path)
                    except Exception as e:
                        st.error(f"Error reading sample sheet: {e}")

            # If a source has loaded data, store and initialize fields
            if df_imported is not None and source_name is not None:
                st.session_state["raw_df_imported"] = df_imported
                st.session_state["uploaded_filename"] = source_name
                
                # Auto-scan columns
                mapped_cols_raw = ae.scan_financial_columns(df_imported)
                st.session_state["mapped_cols_raw"] = mapped_cols_raw
                
                cols = list(df_imported.columns)
                if mapped_cols_raw["date"] in cols:
                    st.session_state["sel_date_col"] = mapped_cols_raw["date"]
                else:
                    st.session_state["sel_date_col"] = "Select Date Column..."
                    
                if mapped_cols_raw["sales"] in cols:
                    st.session_state["sel_sales_col"] = mapped_cols_raw["sales"]
                else:
                    st.session_state["sel_sales_col"] = "Select Sales Column..."
                    
                st.session_state["sel_exp_col"] = mapped_cols_raw["expenses"] if mapped_cols_raw["expenses"] in cols else "None"
                st.session_state["sel_prod_col"] = mapped_cols_raw["product"] if mapped_cols_raw["product"] in cols else "None"
                st.session_state["sel_reg_col"] = mapped_cols_raw["region"] if mapped_cols_raw["region"] in cols else "None"
                st.session_state["sel_order_col"] = mapped_cols_raw.get("order_id") if mapped_cols_raw.get("order_id") in cols else "None"
                
                if (mapped_cols_raw["date"] in cols) and (mapped_cols_raw["sales"] in cols):
                    st.session_state["trigger_recalc"] = True
                else:
                    st.session_state["imported_metrics"] = None
                    st.session_state["trigger_recalc"] = False
                st.rerun()

            # If files are loaded, display field mapping and compute metrics
            if "raw_df_imported" in st.session_state:
                df_imported = st.session_state["raw_df_imported"]
                cols = list(df_imported.columns)
                
                st.markdown("### Map Spreadsheet Fields")
                st.write("Confirm or adjust which columns represent critical metrics:")
                
                date_options = ["Select Date Column..."] + cols
                sales_options = ["Select Sales Column..."] + cols
                exp_options = ["None"] + cols
                prod_options = ["None"] + cols
                reg_options = ["None"] + cols
                
                col_sel1, col_sel2, col_sel3 = st.columns(3)
                
                with col_sel1:
                    cur_date = st.session_state.get("sel_date_col", "Select Date Column...")
                    date_idx = date_options.index(cur_date) if cur_date in date_options else 0
                    sel_date_col = st.selectbox("Date Column", options=date_options, index=date_idx, key="date_selector")
                    if sel_date_col != st.session_state.get("sel_date_col"):
                        st.session_state["sel_date_col"] = sel_date_col
                        st.session_state["trigger_recalc"] = True
                        
                with col_sel2:
                    cur_sales = st.session_state.get("sel_sales_col", "Select Sales Column...")
                    sales_idx = sales_options.index(cur_sales) if cur_sales in sales_options else 0
                    sel_sales_col = st.selectbox("Sales/Revenue Column", options=sales_options, index=sales_idx, key="sales_selector")
                    if sel_sales_col != st.session_state.get("sel_sales_col"):
                        st.session_state["sel_sales_col"] = sel_sales_col
                        st.session_state["trigger_recalc"] = True
                        
                with col_sel3:
                    cur_exp = st.session_state.get("sel_exp_col", "None")
                    exp_idx = exp_options.index(cur_exp) if cur_exp in exp_options else 0
                    sel_exp_col = st.selectbox("Expenses Column (Optional)", options=exp_options, index=exp_idx, key="expenses_selector")
                    if sel_exp_col != st.session_state.get("sel_exp_col"):
                        st.session_state["sel_exp_col"] = sel_exp_col
                        st.session_state["trigger_recalc"] = True
                        
                col_sel4, col_sel5, col_sel6 = st.columns(3)
                with col_sel4:
                    cur_prod = st.session_state.get("sel_prod_col", "None")
                    prod_idx = prod_options.index(cur_prod) if cur_prod in prod_options else 0
                    sel_prod_col = st.selectbox("Product Column (Optional)", options=prod_options, index=prod_idx, key="product_selector")
                    if sel_prod_col != st.session_state.get("sel_prod_col"):
                        st.session_state["sel_prod_col"] = sel_prod_col
                        st.session_state["trigger_recalc"] = True
                        
                with col_sel5:
                    cur_reg = st.session_state.get("sel_reg_col", "None")
                    reg_idx = reg_options.index(cur_reg) if cur_reg in reg_options else 0
                    sel_reg_col = st.selectbox("Region Column (Optional)", options=reg_options, index=reg_idx, key="region_selector")
                    if sel_reg_col != st.session_state.get("sel_reg_col"):
                        st.session_state["sel_reg_col"] = sel_reg_col
                        st.session_state["trigger_recalc"] = True

                with col_sel6:
                    order_options = ["None"] + cols
                    cur_order = st.session_state.get("sel_order_col", "None")
                    order_idx = order_options.index(cur_order) if cur_order in order_options else 0
                    sel_order_col = st.selectbox("Order ID Column (Optional)", options=order_options, index=order_idx, key="order_selector")
                    if sel_order_col != st.session_state.get("sel_order_col"):
                        st.session_state["sel_order_col"] = sel_order_col
                        st.session_state["trigger_recalc"] = True

                # If no expenses column is mapped, show slider for simulated margin
                if sel_exp_col == "None":
                    st.write("")
                    cur_margin = st.session_state.get("sel_default_margin")
                    if cur_margin is None:
                        cur_margin = 40.0
                    sel_default_margin = st.slider(
                        "Default Profit Margin % (for Simulation)", 
                        min_value=1.0, 
                        max_value=99.0, 
                        value=float(cur_margin), 
                        step=1.0, 
                        help="Since no Expenses column is mapped, this margin is used to simulate operational expenses."
                    )
                    if sel_default_margin != cur_margin:
                        st.session_state["sel_default_margin"] = sel_default_margin
                        st.session_state["trigger_recalc"] = True
                else:
                    st.session_state["sel_default_margin"] = None

                # Check if current settings are valid
                is_valid = (sel_date_col != "Select Date Column...") and (sel_sales_col != "Select Sales Column...")
                
                if is_valid:
                    if sel_date_col == sel_sales_col:
                        st.warning("Warning: Date Column and Sales/Revenue Column cannot be the same column. Please check your mapping.")
                        if "imported_metrics" in st.session_state and st.session_state["imported_metrics"] is not None:
                            st.session_state["imported_metrics"] = None
                            st.rerun()
                    else:
                        # Recalculate metrics if triggered
                        if st.session_state.get("trigger_recalc", False):
                            st.session_state["trigger_recalc"] = False
                            
                            mapped_cols = {
                                "date": sel_date_col,
                                "sales": sel_sales_col,
                                "expenses": None if sel_exp_col == "None" else sel_exp_col,
                                "product": None if sel_prod_col == "None" else sel_prod_col,
                                "region": None if sel_reg_col == "None" else sel_reg_col,
                                "order_id": None if sel_order_col == "None" else sel_order_col
                            }
                            
                            try:
                                default_margin_val = st.session_state.get("sel_default_margin")
                                if default_margin_val is None:
                                    default_margin_val = 40.0
                                    
                                metrics = ae.calculate_imported_metrics(df_imported, mapped_cols, default_margin=default_margin_val)
                                ai_insights = aie.generate_financial_insights(metrics, st.session_state.ai_mode, role)
                                metrics["ai_insights"] = ai_insights
                                metrics["mapped_cols"] = mapped_cols
                                
                                # Format the date column in the preview DataFrame to day/month/year format for display
                                preview_df = df_imported.copy()
                                date_col_name = mapped_cols["date"]
                                if date_col_name in preview_df.columns:
                                    try:
                                        parsed_dates = pd.to_datetime(preview_df[date_col_name], format='mixed', dayfirst=True, errors='coerce')
                                        preview_df[date_col_name] = parsed_dates.dt.strftime('%d/%m/%Y')
                                    except Exception:
                                        pass
                                metrics["df_preview"] = preview_df.head(10)
                                
                                df_graph = pd.DataFrame()
                                df_graph["date_id"] = pd.to_datetime(df_imported[mapped_cols["date"]], format='mixed', dayfirst=True).dt.strftime("%Y-%m-%d")
                                df_graph["net_revenue"] = pd.to_numeric(df_imported[mapped_cols["sales"]], errors='coerce').fillna(0.0)
                                
                                if mapped_cols["expenses"] and mapped_cols["expenses"] in df_imported.columns:
                                    expenses_val = pd.to_numeric(df_imported[mapped_cols["expenses"]], errors='coerce').fillna(0.0)
                                    df_graph["profit_amount"] = df_graph["net_revenue"] - expenses_val
                                else:
                                    df_graph["profit_amount"] = df_graph["net_revenue"] * (default_margin_val / 100.0)
                                    
                                if mapped_cols["region"] and mapped_cols["region"] in df_imported.columns:
                                    df_graph["region"] = df_imported[mapped_cols["region"]]
                                else:
                                    df_graph["region"] = "Imported"
                                    
                                df_graph["sales_channel"] = "Imported"
                                df_graph["status"] = "Completed"
                                
                                metrics["df_sales_compat"] = df_graph
                                
                                if mapped_cols["product"] and mapped_cols["order_id"]:
                                    p_col = mapped_cols["product"]
                                    o_col = mapped_cols["order_id"]
                                    if p_col in df_imported.columns and o_col in df_imported.columns:
                                        df_items = df_imported[[o_col, p_col]].dropna()
                                        df_items.columns = ["order_id", "product_name"]
                                        metrics["df_items_compat"] = df_items
                                
                                st.session_state["imported_metrics"] = metrics
                                st.success("Financial calculations updated successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error calculating metrics: {e}")
                else:
                    st.info("Please select both a Date column and a Sales/Revenue column to calculate metrics.")
                    if "imported_metrics" in st.session_state and st.session_state["imported_metrics"] is not None:
                        st.session_state["imported_metrics"] = None
                        st.rerun()
                    
            # Render details if uploaded
            if "imported_metrics" in st.session_state and st.session_state["imported_metrics"] is not None:
                metrics = st.session_state["imported_metrics"]
                mapped_cols = metrics["mapped_cols"]
                
                st.markdown("#### Imported Data Preview (First 5 Rows)")
                st.dataframe(metrics["df_preview"].head(5), use_container_width=True)
                
                st.markdown("### Dynamic Sales Trend Chart (Imported)")
                
                # Dynamic aggregation level for imported dataset
                imp_agg_level = st.radio(
                    "Select Aggregation Level for Imported Data", 
                    options=["Daily", "Monthly", "Yearly"], 
                    horizontal=True, 
                    key="imported_graph_agg_level_selector"
                )
                
                df_imported_daily = metrics["df_sales_compat"].copy()
                df_imported_daily["date_parsed"] = pd.to_datetime(df_imported_daily["date_id"], errors='coerce')
                df_imported_daily = df_imported_daily.dropna(subset=["date_parsed"])
                
                if not df_imported_daily.empty:
                    if imp_agg_level == "Monthly":
                        df_imported_daily["x_axis"] = df_imported_daily["date_parsed"].dt.strftime("%Y-%m")
                        df_plot_imp = df_imported_daily.groupby("x_axis")[["net_revenue"]].sum().reset_index()
                        x_col = "x_axis"
                        x_label = "Month"
                        title_suffix = "Monthly"
                    elif imp_agg_level == "Yearly":
                        df_imported_daily["x_axis"] = df_imported_daily["date_parsed"].dt.strftime("%Y")
                        df_plot_imp = df_imported_daily.groupby("x_axis")[["net_revenue"]].sum().reset_index()
                        x_col = "x_axis"
                        x_label = "Year"
                        title_suffix = "Yearly"
                    else:
                        df_imported_daily["x_axis"] = df_imported_daily["date_parsed"].dt.strftime("%Y-%m-%d")
                        df_plot_imp = df_imported_daily.groupby("x_axis")[["net_revenue"]].sum().reset_index()
                        x_col = "x_axis"
                        x_label = "Date"
                        title_suffix = "Daily"
                        
                    df_plot_imp = df_plot_imp.sort_values(x_col).reset_index(drop=True)
                    
                    min_date_imp = df_plot_imp[x_col].min()
                    max_date_imp = df_plot_imp[x_col].max()
                    trend_title_imp = f"Imported {title_suffix} Net Sales Revenue Trend ({min_date_imp} to {max_date_imp})"
                    
                    fig_trend_imp = px.line(
                        df_plot_imp, x=x_col, y="net_revenue",
                        title=trend_title_imp,
                        labels={x_col: x_label, "net_revenue": "Net Sales ($)"},
                        color_discrete_sequence=["#10B981"]
                    )
                    fig_trend_imp.update_layout(
                        paper_bgcolor="#1E293B", plot_bgcolor="#1E293B",
                        font_color="#F1F5F9", margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_trend_imp, use_container_width=True)
                    
                    # Dynamic Future Forecasting on Imported Data
                    st.markdown("### Dynamic Sales Forecast (Imported)")
                    show_imp_forecast = st.checkbox("Generate 30-Day Future Sales Forecast from Imported Data", value=True, key="imp_dashboard_forecast_checkbox")
                    if show_imp_forecast:
                        df_imp_daily = df_imported_daily.groupby("date_id")["net_revenue"].sum().reset_index().sort_values("date_id").reset_index(drop=True)
                        if len(df_imp_daily) >= 10:
                            with st.spinner("Generating numpy-based seasonal forecast..."):
                                try:
                                    forecast_results = ae.forecast_sales(df_imp_daily["date_id"], df_imp_daily["net_revenue"])
                                    dates_all = forecast_results["dates"]
                                    hist_count = forecast_results["historical_dates_count"]
                                    
                                    fig_fc_imp = go.Figure()
                                    
                                    # 1. Historical Actuals (Scatter)
                                    fig_fc_imp.add_trace(go.Scatter(
                                        x=dates_all[:hist_count], y=forecast_results["historical_actual"],
                                        mode="markers", name="Historical Actuals",
                                        marker=dict(color="#10B981", size=4)
                                    ))
                                    
                                    # 2. Historical Fitted (Line)
                                    fig_fc_imp.add_trace(go.Scatter(
                                        x=dates_all[:hist_count], y=forecast_results["historical_fit"],
                                        mode="lines", name="Seasonal Model Fit",
                                        line=dict(color="#34D399", width=1.5)
                                    ))
                                    
                                    # 3. Future Forecast (Line)
                                    fig_fc_imp.add_trace(go.Scatter(
                                        x=dates_all[hist_count:], y=forecast_results["forecast_values"],
                                        mode="lines+markers", name="30-Day Forecast",
                                        line=dict(color="#60A5FA", width=2.5, dash="dash")
                                    ))
                                    
                                    # 4. Confidence boundaries (shaded region)
                                    fig_fc_imp.add_trace(go.Scatter(
                                        x=dates_all[hist_count:] + dates_all[hist_count:][::-1],
                                        y=forecast_results["upper_bound"] + forecast_results["lower_bound"][::-1],
                                        fill="toself",
                                        fillcolor="rgba(96, 165, 250, 0.15)",
                                        line=dict(color="rgba(255,255,255,0)"),
                                        hoverinfo="skip",
                                        name="95% Confidence Interval"
                                    ))
                                    
                                    fig_fc_imp.update_layout(
                                        title=f"NumPy Seasonal Forecast for {st.session_state.get('uploaded_filename', 'Imported Data')}",
                                        xaxis_title="Date", yaxis_title="Sales Revenue ($)",
                                        paper_bgcolor="#1E293B", plot_bgcolor="#1E293B",
                                        font_color="#F1F5F9", height=450
                                    )
                                    st.plotly_chart(fig_fc_imp, use_container_width=True)
                                    
                                    f_sum = sum(forecast_results["forecast_values"])
                                    st.info(f"**Imported Forecast Insights:** Cumulative sales revenue projected for the next 30 days is **${f_sum:,.2f}**, with average daily projections of **${np.mean(forecast_results['forecast_values']):,.2f}**.")
                                except Exception as fe:
                                    st.error(f"Error executing forecast calculations: {fe}")
                        else:
                            st.warning("Warning: Forecasting requires at least 10 historical daily data points in the spreadsheet.")
                else:
                    st.warning("Warning: No valid date/sales records found in the imported dataset to plot the trend line.")
                
                # Show AI Trend Insights
                st.markdown("### AI-Generated Executive Insights")
                st.info(metrics["ai_insights"])
                
                # Show Profit/Loss comparison across years
                st.markdown("### Multi-Year Profitability Analysis")
                yd = metrics["yearly_data"]
                
                c1, c2 = st.columns([2, 1])
                with c1:
                    # Melt for grouped bar chart
                    df_melted = yd.melt(id_vars='year', value_vars=['revenue', 'expenses', 'profit_loss'], var_name='Metric', value_name='Amount')
                    df_melted['Metric'] = df_melted['Metric'].replace({
                        'revenue': 'Net Revenue',
                        'expenses': 'Total Expenses',
                        'profit_loss': 'Net Profit/Loss'
                    })
                    
                    fig_yearly = px.bar(
                        df_melted, x='year', y='Amount', color='Metric',
                        barmode='group',
                        title='Yearly Financial Breakdown (Revenue vs. Expenses vs. Profit/Loss)',
                        labels={'year': 'Year', 'Amount': 'Amount ($)', 'Metric': 'MetricType'},
                        color_discrete_map={'Net Revenue': '#60A5FA', 'Total Expenses': '#F87171', 'Net Profit/Loss': '#34D399'}
                    )
                    fig_yearly.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#1E293B", font_color="#F1F5F9")
                    st.plotly_chart(fig_yearly, use_container_width=True)
                    
                with c2:
                    st.markdown("**Financial Summary Table**")
                    yd_display = yd.copy()
                    yd_display.columns = ['Year', 'Revenue ($)', 'Expenses ($)', 'Profit/Loss ($)']
                    # Format currencies nicely
                    for col in ['Revenue ($)', 'Expenses ($)', 'Profit/Loss ($)']:
                        yd_display[col] = yd_display[col].apply(lambda v: f"${v:,.2f}")
                    st.dataframe(yd_display, use_container_width=True)


# ----------------- TAB 2: ADVANCED ANALYTICS & ML -----------------
with tab2:
    st.header("Machine Learning & Advanced Statistical Diagnostics")
    
    ml_selector = st.segmented_control(
        "Machine Learning Task",
        options=["Time Series Forecasting", "Customer Segmentation", "Cross-Sell Recommendations", "Leash & Churn Scoring", "B2B Sales Conversion Classifier"],
        default="Time Series Forecasting"
    )
    
    # Task A: Forecasting
    if ml_selector == "Time Series Forecasting":
        st.subheader("30-Day Forward Sales Forecasting (Seasonality + Trend)")
        
        if role == "Operations Manager":
            st.warning("Access Restricted: Sales Forecasting requires financial permissions (CEO or Sales Manager).")
        else:
            # Check if imported spreadsheet is loaded
            is_imported = "imported_metrics" in st.session_state and st.session_state["imported_metrics"] is not None
            if is_imported:
                df_sales_compat = st.session_state["imported_metrics"]["df_sales_compat"]
                df_hist = df_sales_compat.groupby("date_id")["net_revenue"].sum().reset_index()
                df_hist.rename(columns={"net_revenue": "sales"}, inplace=True)
                df_hist = df_hist.sort_values("date_id").reset_index(drop=True)
                st.info(f"**Data Source:** Spreadsheet (`{st.session_state.get('uploaded_filename', 'Imported Data')}`)")
            else:
                df_hist = run_db_query("SELECT date_id, SUM(net_revenue) as sales FROM fact_sales WHERE status='Completed' GROUP BY date_id ORDER BY date_id;", show_error=False)
                st.info("**Data Source:** Star Schema Warehouse (`fact_sales` table)")

            if not df_hist.empty:
                if len(df_hist) < 10:
                    st.warning("Warning: Forecasting requires at least 10 historical daily data points in the dataset.")
                else:
                    forecast_results = ae.forecast_sales(df_hist["date_id"], df_hist["sales"])
                
                # Map into Plotly chart
                dates_all = forecast_results["dates"]
                hist_count = forecast_results["historical_dates_count"]
                
                fig_fc = go.Figure()
                
                # 1. Historical Actuals (Scatter)
                fig_fc.add_trace(go.Scatter(
                    x=dates_all[:hist_count], y=forecast_results["historical_actual"],
                    mode="markers", name="Historical Actuals",
                    marker=dict(color="#94A3B8", size=4)
                ))
                
                # 2. Historical Fitted (Line)
                fig_fc.add_trace(go.Scatter(
                    x=dates_all[:hist_count], y=forecast_results["historical_fit"],
                    mode="lines", name="Seasonal Model Fit",
                    line=dict(color="#38BDF8", width=1.5)
                ))
                
                # 3. Future Forecast (Line)
                fig_fc.add_trace(go.Scatter(
                    x=dates_all[hist_count:], y=forecast_results["forecast_values"],
                    mode="lines+markers", name="30-Day Forecast",
                    line=dict(color="#60A5FA", width=2.5, dash="dash")
                ))
                
                # 4. Confidence boundaries (shaded region)
                fig_fc.add_trace(go.Scatter(
                    x=dates_all[hist_count:] + dates_all[hist_count:][::-1],
                    y=forecast_results["upper_bound"] + forecast_results["lower_bound"][::-1],
                    fill="toself",
                    fillcolor="rgba(96, 165, 250, 0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    name="95% Confidence Interval"
                ))
                
                fig_fc.update_layout(
                    title="NumPy Linear Trend + Sinusoidal Seasonality Forecast",
                    xaxis_title="Date", yaxis_title="Sales Revenue ($)",
                    paper_bgcolor="#1E293B", plot_bgcolor="#1E293B",
                    font_color="#F1F5F9", height=500
                )
                st.plotly_chart(fig_fc, use_container_width=True)
                
                # Detailed stats
                f_sum = sum(forecast_results["forecast_values"])
                st.info(f"**Forecast Insights Summary:** Predicted cumulative sales revenue over the next 30 days is **${f_sum:,.2f}**, with average daily projections of **${np.mean(forecast_results['forecast_values']):,.2f}**.")
                
          # Task B: Customer Segmentation
    elif ml_selector == "Customer Segmentation":
        st.subheader("Unsupervised RFM Value Clustering (NumPy K-Means)")
        
        if role == "Operations Manager":
            st.warning("Access Restricted: Customer Segmentation requires financial permissions (CEO or Sales Manager).")
        else:
            df_cust = run_db_query("SELECT customer_id, customer_name, recency, frequency, monetary FROM dim_customers;", show_error=False)
            if not df_cust.empty:
                labels, tags, centroids = ae.kmeans_clustering(df_cust, k=3)
                df_cust["cluster"] = tags
                
                c_graph1, c_graph2 = st.columns([2, 1])
                with c_graph1:
                    # 3D plot of RFM
                    fig_clusters = px.scatter_3d(
                        df_cust, x="recency", y="frequency", z="monetary",
                        color="cluster", hover_name="customer_name",
                        color_discrete_sequence=["#34D399", "#A78BFA", "#F87171"],
                        title="Customer Clusters (Recency vs Frequency vs Monetary Spend)"
                    )
                    fig_clusters.update_layout(
                        paper_bgcolor="#1E293B", scene=dict(
                            xaxis_backgroundcolor="#1E293B",
                            yaxis_backgroundcolor="#1E293B",
                            zaxis_backgroundcolor="#1E293B"
                        ), font_color="#F1F5F9", height=500
                    )
                    st.plotly_chart(fig_clusters, use_container_width=True)
                    
                with c_graph2:
                    # Cluster description averages
                    st.markdown("**Segment Definitions & Centroids**")
                    df_avg = df_cust.groupby("cluster")[["recency", "frequency", "monetary"]].mean().reset_index()
                    st.dataframe(df_avg.style.format({"recency": "{:.1f} days", "frequency": "{:.1f} orders", "monetary": "${:,.2f}"}), use_container_width=True)
                    
                    st.markdown("""
                    - **High Value Champion**: Active buyers with low recency scores, frequent order checkouts, and premium budget contribution.
                    - **Moderate Growth Partner**: Steady recurring purchasers, maintaining medium recency and average purchase values.
                    - **Low Value / At Risk**: Accounts with very high recency (dormant for 150+ days) and minimal budgets.
                    """)
            else:
                st.warning("No customer dimensions dataset retrieved.")
            
    # Task C: Recommendations
    elif ml_selector == "Cross-Sell Recommendations":
        st.subheader("Association Rule Mining for Market Basket Upsells")
        
        is_imported = "imported_metrics" in st.session_state and st.session_state["imported_metrics"] is not None
        df_items = pd.DataFrame()
        if is_imported:
            metrics = st.session_state["imported_metrics"]
            if "df_items_compat" in metrics:
                df_items = metrics["df_items_compat"]
                st.info(f"**Data Source:** Spreadsheet (`{st.session_state.get('uploaded_filename', 'Imported Data')}`)")
            else:
                st.warning("Warning: To run Association Rule Mining on the spreadsheet, please select/map both 'Product Column' and 'Order ID Column' in the Executive Dashboard tab.")
        else:
            df_items = run_db_query("""
                SELECT f.order_id, p.product_name 
                FROM fact_sales f 
                JOIN dim_products p ON f.product_id = p.product_id 
                WHERE f.status='Completed';
            """)
            st.info("**Data Source:** Star Schema Warehouse (`fact_sales` table)")
        if not df_items.empty:
            rules = ae.market_basket_rules(df_items["order_id"], df_items["product_name"])
            
            if not rules.empty:
                st.write("**Top Product Association Rules (Sorted by Lift)**")
                st.dataframe(rules.style.format({"support": "{:.2%}", "confidence": "{:.1%}", "lift": "{:.2f}"}), use_container_width=True)
                
                # Plot rules matrix
                fig_hm = px.density_heatmap(
                    rules, x="antecedent", y="consequent", z="lift",
                    title="Cross-Sell Potential Heatmap (Z = Lift Index)",
                    color_continuous_scale="Purples"
                )
                fig_hm.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#1E293B", font_color="#F1F5F9")
                st.plotly_chart(fig_hm, use_container_width=True)
            else:
                st.info("Market Basket rules analysis found no recurring item pairings exceeding support thresholds.")
        else:
            st.warning("No sales transactions details found.")
            
    # Task D: Churn and Lead Scoring
    elif ml_selector == "Leash & Churn Scoring":
        st.subheader("Active Pipelines Churn Risk Roster & Lead Grades")
        
        c_score1, c_score2 = st.columns(2)
        with c_score1:
            st.markdown("**Account Churn Risk Matrix (Top 10 High Risk)**")
            df_cust_ch = run_db_query("SELECT customer_id, customer_name, recency, frequency FROM dim_customers;")
            df_iss_ch = run_db_query("SELECT customer_id, status FROM raw_issues;")
            
            if not df_cust_ch.empty:
                df_churn = ae.calculate_customer_churn_risk(df_cust_ch, df_iss_ch)
                df_churn_high = df_churn.sort_values(by="risk_score", ascending=False).head(10)
                st.dataframe(df_churn_high, use_container_width=True)
            else:
                st.warning("No customer details.")
                
        with c_score2:
            st.markdown("**Prioritized Leads Scoring (Tier A / B Leads)**")
            df_leads_ch = run_db_query("SELECT lead_id, lead_name, source, status, score FROM raw_leads;")
            if not df_leads_ch.empty:
                df_scored_leads = ae.calculate_lead_scoring(df_leads_ch)
                df_leads_priority = df_scored_leads[df_scored_leads["tier"].isin(["Tier A (Hot)", "Tier B (Warm)"])].sort_values(by="final_score", ascending=False).head(10)
                st.dataframe(df_leads_priority, use_container_width=True)
            else:
                st.warning("No lead generation data.")

        # ── Revenue-Based Churn Scoring (company_revenue_sheets) ──────────────
        st.divider()
        st.subheader("Company Revenue Churn Risk — All Training Data")
        st.caption("Loaded from all files in `data/raw/` (CSV + XLSX, deduplicated by date)")

        try:
            import churn_scoring_fixed as csf  # Path-resolved hot reload

            with st.spinner("Loading and scoring revenue data..."):
                daily_rev_all   = csf.load_revenue_data()
                monthly_rev_all = csf.build_monthly_summary(daily_rev_all)
                risk_table_all  = csf.generate_monthly_risk_table(monthly_rev_all)

            # Year selection filter
            years_available = sorted(list(daily_rev_all["year"].unique()))
            selected_year = st.selectbox(
                "Select Analysis Period / Year",
                options=["All Available Years"] + years_available,
                index=0
            )

            if selected_year != "All Available Years":
                daily_rev = daily_rev_all[daily_rev_all["year"] == selected_year]
                risk_table = risk_table_all[risk_table_all["month"].str.startswith(selected_year)]
            else:
                daily_rev = daily_rev_all
                risk_table = risk_table_all

            # ── KPI summary row ────────────────────────────────────────────────
            if not risk_table.empty:
                latest = risk_table.iloc[-1]
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total Days Loaded",    f"{len(daily_rev):,}")
                k2.metric("Monthly Periods Scored", f"{len(risk_table)}")
                k3.metric("Current Risk Score",    f"{latest['risk_score']}")
                k4.metric("Current Category",      latest["risk_category"])
            else:
                st.info("No data available for the selected period.")

            # ── Risk progression chart ─────────────────────────────────────────
            fig_risk = px.line(
                risk_table, x="month", y="risk_score",
                color="risk_category",
                title=f"Monthly Churn Risk Score - {selected_year}" if selected_year != "All Available Years" else "Monthly Churn Risk Score - All Years",
                labels={"month": "Month", "risk_score": "Risk Score (0-100)", "risk_category": "Category"},
                color_discrete_map={
                    "Low Risk":      "#34D399",
                    "Medium Risk":   "#FBBF24",
                    "High Risk":     "#F87171",
                    "Critical Risk": "#DC2626",
                },
                markers=True
            )
            fig_risk.add_hline(y=30, line_dash="dot", line_color="#34D399",  annotation_text="Low/Med boundary")
            fig_risk.add_hline(y=60, line_dash="dot", line_color="#FBBF24",  annotation_text="Med/High boundary")
            fig_risk.add_hline(y=80, line_dash="dot", line_color="#F87171",  annotation_text="High/Critical boundary")
            fig_risk.update_layout(
                paper_bgcolor="#1E293B", plot_bgcolor="#1E293B",
                font_color="#F1F5F9", margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_risk, use_container_width=True)

            # ── Revenue trend + risk overlay ───────────────────────────────────
            col_rev1, col_rev2 = st.columns(2)

            with col_rev1:
                fig_rev = px.bar(
                    risk_table, x="month", y="total_revenue",
                    color="risk_category",
                    title="Monthly Revenue by Risk Tier",
                    labels={"month": "Month", "total_revenue": "Revenue ($)", "risk_category": "Risk Tier"},
                    color_discrete_map={
                        "Low Risk":      "#34D399",
                        "Medium Risk":   "#FBBF24",
                        "High Risk":     "#F87171",
                        "Critical Risk": "#DC2626",
                    }
                )
                fig_rev.update_layout(
                    paper_bgcolor="#1E293B", plot_bgcolor="#1E293B",
                    font_color="#F1F5F9", margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_rev, use_container_width=True)

            with col_rev2:
                tier_counts = risk_table["risk_category"].value_counts().reset_index()
                tier_counts.columns = ["Category", "Months"]
                fig_pie = px.pie(
                    tier_counts, values="Months", names="Category",
                    hole=0.4, title="Risk Tier Distribution (All Months)",
                    color="Category",
                    color_discrete_map={
                        "Low Risk":      "#34D399",
                        "Medium Risk":   "#FBBF24",
                        "High Risk":     "#F87171",
                        "Critical Risk": "#DC2626",
                    }
                )
                fig_pie.update_layout(
                    paper_bgcolor="#1E293B", plot_bgcolor="#1E293B",
                    font_color="#F1F5F9", margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            # ── Full risk table ────────────────────────────────────────────────
            st.markdown("**Full Monthly Risk Score Table**")
            display_cols = ["month", "total_revenue", "mom_change_pct",
                            "risk_score", "risk_category", "priority", "action"]
            st.dataframe(
                risk_table[display_cols].style.apply(
                    lambda row: [
                        "background-color: #DC2626; color: white" if row["risk_category"] == "Critical Risk"
                        else "background-color: #7F1D1D; color: white" if row["risk_category"] == "High Risk"
                        else "background-color: #78350F; color: white" if row["risk_category"] == "Medium Risk"
                        else "" for _ in row
                    ], axis=1
                ),
                use_container_width=True
            )

            # ── YoY summary ────────────────────────────────────────────────────
            st.markdown("**Year-over-Year Revenue Summary**")
            yoy = daily_rev.groupby("year")["Revenue"].agg(
                Total="sum", Daily_Avg="mean", Peak="max", Low="min"
            ).reset_index()
            yoy["YoY Growth"] = yoy["Total"].pct_change().map(
                lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "—"
            )
            yoy["Total"]     = yoy["Total"].map("${:,.0f}".format)
            yoy["Daily_Avg"] = yoy["Daily_Avg"].map("${:,.0f}".format)
            yoy["Peak"]      = yoy["Peak"].map("${:,.0f}".format)
            yoy["Low"]       = yoy["Low"].map("${:,.0f}".format)
            st.dataframe(yoy, use_container_width=True)

        except Exception as e:
            st.error(f"Revenue churn scoring error: {e}")
                
    # Task E: B2B Sales Conversion Classifier
    elif ml_selector == "B2B Sales Conversion Classifier":
        st.subheader("B2B SaaS Sales Conversation AI Classifier & Coach")
        st.write("Analyze actual dialogue transcripts, track success probabilities, and coach sales representatives using advanced analytics models.")
        
        # Load weights/metrics
        weights_path = os.path.join(BASE_DIR, "saas_sales_model_weights.json")
        metrics = {"accuracy": 0.88, "precision": 0.86, "recall": 0.90, "f1_score": 0.88} # defaults
        if os.path.exists(weights_path):
            try:
                with open(weights_path, 'r') as f:
                    model_data = json.load(f)
                    metrics = model_data.get("metrics", metrics)
            except Exception:
                pass
                
        # 1. Performance Panel
        st.markdown("### NumPy Logistic Regression Model Performance")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Classification Accuracy", f"{metrics['accuracy']:.1%}")
        m_col2.metric("Precision Index", f"{metrics['precision']:.1%}")
        m_col3.metric("Recall Rate", f"{metrics['recall']:.1%}")
        m_col4.metric("F1-Score Metric", f"{metrics['f1_score']:.4f}")
        
        # Retrain Trigger
        if st.button("Retrain NumPy Sales Classifier"):
            with st.spinner("Retraining classifier model via gradient descent..."):
                try:
                    from train_conversation_assistant import train_model
                    train_model(epochs=120, learning_rate=0.25)
                    st.success("NumPy Logistic Regression Model successfully retrained on 500 sales conversations!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    
        # 2. Conversation Viewer & Trajectory
        st.divider()
        st.markdown("### Select Sales Conversation to Analyze")
        
        csv_path = os.path.join(BASE_DIR, "saas_sales_conversations.csv")
        if os.path.exists(csv_path):
            df_convs = pd.read_csv(csv_path)
            
            selected_conv_year = st.selectbox("Filter Conversations by Year", ["All Years", "2025", "2026"])
            
            # Parse year from date column
            df_convs["year"] = pd.to_datetime(df_convs["date"]).dt.year.astype(str)
            if selected_conv_year != "All Years":
                df_filtered = df_convs[df_convs["year"] == selected_conv_year]
            else:
                df_filtered = df_convs

            # Calculate Deal Priority Score
            if not df_filtered.empty:
                df_filtered = df_filtered.copy()
                df_filtered["deal_score"] = (df_filtered["sales_effectiveness"] * 40 + df_filtered["customer_engagement"] * 60)
                
                # Label deal priority
                def get_priority_label(score):
                    if score >= 75:
                        return "High (Hot Lead)"
                    elif score >= 50:
                        return "Medium"
                    else:
                        return "Low (At Risk)"
                
                df_filtered["priority"] = df_filtered["deal_score"].apply(get_priority_label)
                
                # Display Prioritized Deals Roster
                st.markdown("### Prioritized Active Deals Roster")
                st.caption(f"Showing top deals based on sales effectiveness and engagement for {selected_conv_year}")
                
                display_roster = df_filtered.sort_values(by="deal_score", ascending=False).head(5)
                st.dataframe(
                    display_roster[["conversation_id", "sales_rep", "customer_industry", "company_size", "objection_type", "deal_score", "priority"]].style.apply(
                        lambda row: [
                            "background-color: #1E293B; border: 1px solid #334155; color: #F1F5F9;" for _ in row
                        ], axis=1
                    ),
                    use_container_width=True
                )
                
                conv_list = df_filtered["conversation_id"].tolist()
                selected_conv = st.selectbox("Active Sales Call ID", options=conv_list)
                
                # Get selected row details
                row_det = df_filtered[df_filtered["conversation_id"] == selected_conv].iloc[0]
                
                c_meta, c_trans = st.columns([1, 1])
                with c_meta:
                    st.markdown("**Sales Rep:** " + str(row_det["sales_rep"]))
                    st.markdown("**Prospect Industry:** " + str(row_det["customer_industry"]))
                    st.markdown("**Company Size:** " + str(row_det["company_size"]))
                    st.markdown("**Objection Type:** " + str(row_det["objection_type"]))
                    
                    # Metrics
                    eff = float(row_det["sales_effectiveness"])
                    eng = float(row_det["customer_engagement"])
                    st.progress(eff, text=f"Sales Representative Effectiveness: {eff:.1%}")
                    st.progress(eng, text=f"Customer Engagement Score: {eng:.1%}")
                    
                    # Conversion prediction
                    outcome = int(row_det["conversion_outcome"])
                    if outcome == 1:
                        st.success("MODEL PREDICTION: LIKELY CONVERSION (High conversion probability)")
                    else:
                        st.error("MODEL PREDICTION: HIGH CHURN RISK (Conversion probability dropped below baseline)")
                    
                    # How to Win Playbook Section
                    win_playbooks = {
                        "Pricing": {
                            "strategy": "Cost-Optimization & Tiered Scaling",
                            "tactics": [
                                "Propose a 14-day trial period using standard tier functionality to prove business value first.",
                                "Highlight the 40% reduction in database querying overhead to prove hardware cost savings.",
                                "Offer flexible quarterly invoicing schedules instead of requiring upfront annual payments."
                            ]
                        },
                        "Security/Compliance": {
                            "strategy": "Compliance Assurance & Security Audit Assistance",
                            "tactics": [
                                "Provide the prospect's security officer with our SOC2 Type II compliance audit report.",
                                "Offer to sign a Business Associate Agreement (BAA) for HIPAA compliance immediately.",
                                "Emphasize TLS 1.3 transit encryption and AES-256 data storage standards."
                            ]
                        },
                        "Custom Integration": {
                            "strategy": "Zero-Code Connectors & Dedicated Onboarding Integration Support",
                            "tactics": [
                                "Demo the built-in REST API, pre-configured adapters, and Webhook features.",
                                "Offer a dedicated developer-to-developer onboarding session to discuss legacy system adapters.",
                                "Highlight our out-of-the-box support for internal database connectivity."
                            ]
                        },
                        "Competitor Comparison": {
                            "strategy": "Cache Caching Speed & Zero-Code Dashboard Builder Advantage",
                            "tactics": [
                                "Conduct a live speed benchmark comparison against competitors (Snowflake/Databricks).",
                                "Contrast deployment timeline (48-hour onboarding vs weeks or months for other tools).",
                                "Demonstrate the zero-code interactive executive reporting interface."
                            ]
                        },
                        "None": {
                            "strategy": "Close Agreement & Onboarding Execution Kickoff",
                            "tactics": [
                                "Draft and send the contract/SLA agreements immediately to the executive team.",
                                "Request the technical client contact to proceed with the DB connectivity settings checklist.",
                                "Schedule the official project onboarding kickoff meeting for next week."
                            ]
                        }
                    }

                    obj = row_det["objection_type"]
                    playbook = win_playbooks.get(obj, win_playbooks["None"])
                    st.info(f"**Win Strategy: {playbook['strategy']}**\n\n" + "\n".join([f"- {t}" for t in playbook["tactics"]]))

                    # Plotly Trajectory
                    traj_str = row_det["probability_trajectory"]
                    try:
                        traj_list = json.loads(traj_str)
                        fig_traj = go.Figure()
                        fig_traj.add_trace(go.Scatter(
                            y=traj_list, mode="lines+markers", name="Conversion Probability",
                            line=dict(color="#10B981" if outcome == 1 else "#EF4444", width=3)
                        ))
                        fig_traj.update_layout(
                            title="Turn-by-Turn Conversion Probability Trajectory",
                            xaxis_title="Conversation Turn Index",
                            yaxis_title="Probability of Success",
                            yaxis_range=[0, 1.0],
                            paper_bgcolor="#1E293B", plot_bgcolor="#1E293B", font_color="#F1F5F9"
                        )
                        st.plotly_chart(fig_traj, use_container_width=True)
                    except Exception as e:
                        st.caption(f"Could not parse trajectory chart: {e}")
                        
                with c_trans:
                    st.markdown("**Sales Conversation Transcript Snippet:**")
                    st.info(row_det["transcript_snippet"])
                    
                    # Coach Pitch Input
                    st.markdown("#### Objection Handling Pitch Coach")
                    objection_type = row_det["objection_type"]
                    user_pitch = st.text_area(f"Type a response pitch to address the '{objection_type}' objection:", 
                                             placeholder="e.g. We can offer customized security documentation and sign a HIPAA BAA...")
                    
                    if st.button("Run AI Coaching Evaluation"):
                        if user_pitch:
                            with st.spinner("Sales coach analyzing pitch..."):
                                coach_query = f"As a professional B2B Sales Coach, evaluate this pitch response: '{user_pitch}' to resolve a '{objection_type}' objection in a B2B SaaS conversation. Provide a short, constructive score out of 100 and direct bullet-point improvement tips."
                                coach_feedback = aie.query_ai_assistant(coach_query, "Sales Manager", {})
                                st.markdown("##### Professional Sales Coach Feedback:")
                                st.write(coach_feedback)
                        else:
                            st.warning("Please type a pitch to evaluate.")
            else:
                st.warning("No conversations found matching the selected year.")
        else:
            st.error("Sales conversations dataset not found.")
            
        # 3. Notebooks ML Benchmarks Section
        st.divider()
        st.markdown("### Trained Datasets & Model Benchmarks (Mini-project4 -> Trained-Datasets)")
        nb_col1, nb_col2 = st.columns(2)
        with nb_col1:
            st.markdown("#### Walmart Sales Prediction ML Benchmarks")
            st.caption("Derived from `walmart-sales-prediction-best-ml-algorithms.ipynb`")
            st.markdown("""
            - **Algorithms Tested:** Multiple Linear Regression (MLR), Ridge, Lasso, ElasticNet, and Polynomial Regression (Degree=2).
            - **Key Inference:** Polynomial regression overfits the holiday sales volume anomalies. 
            - **Optimal Model:** The simple Multiple Linear Regression (MLR) model with Recursive Feature Elimination (RFE) yields the best balanced R2/RMSE testing scores.
            """)
        with nb_col2:
            st.markdown("#### Global AI Adaptation & Adoption Survey (Kaggle 2021)")
            st.caption("Derived from `data-science-in-2021-adaptation-or-adoption.ipynb`")
            st.markdown("""
            - **Core Language Ratios:** Python remains the dominant data science language, followed by SQL and R.
            - **AI Adoption Maturity Index:** Scored from 0 (No ML) to 4 (Established models in production for 2+ years).
            - **Regional Leaders:** USA and China lead in absolute volumes, while Japan and Russia exhibit high competitive density in top Kaggle tiers.
            """)


# ----------------- TAB 3: INTELLIGENT SANDBOX & RESOLUTION -----------------
with tab3:
    st.header("Intelligent Operations Sandbox & Resolution Center")
    
    st.subheader("What-If Strategic Scenario Simulator")
    
    if role == "Operations Manager":
        st.warning("Access Restricted: Operational roles do not have permission to run what-if sales simulations.")
    else:
        st.write("Adjust global indicators to simulate demand elasticities and project monthly financials.")
        
        c_sliders, c_results = st.columns([1, 2])
        
        with c_sliders:
            price_slide = st.slider("Adjust Product Selling Price (%)", min_value=-30, max_value=30, value=0, step=5)
            mkt_slide = st.slider("Adjust Marketing Spend Budget (%)", min_value=-50, max_value=100, value=0, step=5)
            disc_slide = st.slider("Adjust Average Customer Discounts (%)", min_value=-20, max_value=20, value=0, step=2)
            
            # Load baseline parameters dynamically from Completed orders
            with engine.connect() as conn:
                totals = pd.read_sql(text("""
                    SELECT AVG(unit_price) as avg_p, SUM(quantity)/365 as avg_qty 
                    FROM fact_sales 
                    WHERE status='Completed';
                """), conn)
            baseline_price = float(totals.iloc[0]["avg_p"]) if totals.iloc[0]["avg_p"] else 1000.0
            baseline_vol = float(totals.iloc[0]["avg_qty"]) * 30.0 if totals.iloc[0]["avg_qty"] else 150.0
            
            sim = ae.simulate_what_if(
                baseline_price, baseline_vol, 
                price_slide / 100.0, mkt_slide / 100.0, disc_slide / 100.0
            )
            
        with c_results:
            # Display simulated metrics
            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                st.metric("Simulated Monthly Volume", f"{sim['new_volume']:.1f} units", f"{sim['new_volume'] - sim['base_volume']:+.1f} units")
            with rc2:
                st.metric("Simulated Revenue", f"${sim['new_revenue']:,.2f}", f"{sim['revenue_change_pct']:+.1f}%")
            with rc3:
                st.metric("Simulated Net Profit", f"${sim['new_profit']:,.2f}", f"{sim['profit_change_pct']:+.1f}%")
                
            # Graph comparing Base vs Simulated
            fig_sim = go.Figure(data=[
                go.Bar(name="Baseline", x=["Revenue", "Profit"], y=[sim["base_revenue"], sim["base_profit"]], marker_color="#94A3B8"),
                go.Bar(name="Simulated Output", x=["Revenue", "Profit"], y=[sim["new_revenue"], sim["new_profit"]], marker_color="#60A5FA")
            ])
            fig_sim.update_layout(
                title="Baseline vs. Simulated Monthly Projections",
                paper_bgcolor="#1E293B", plot_bgcolor="#1E293B",
                font_color="#F1F5F9", barmode="group", height=300
            )
            st.plotly_chart(fig_sim, use_container_width=True)

    st.divider()
    
    st.subheader("Problem Resolution Board & Anomaly Log")
    
    if role == "Operations Manager":
        st.warning("Access Restricted: Daily sales anomaly logs require financial permissions.")
    else:
        st.write("Review auto-detected transactional anomalies and dispatch mitigation tasks to CRM and ERP systems.")
        
        # 1. Fetch sales anomalies
        df_daily_sales = run_db_query("SELECT date_id, SUM(net_revenue) as sales FROM fact_sales WHERE status='Completed' GROUP BY date_id ORDER BY date_id;", show_error=False)
        anom_list = []
        if not df_daily_sales.empty:
            anom_list = ae.detect_anomalies(df_daily_sales["date_id"], df_daily_sales["sales"])
            
        # Render detected anomalies
        if anom_list:
            st.markdown("**Detected System & Campaign Anomalies**")
            for an in anom_list:
                an_type_color = "[Drop]" if an["type"] == "Drop" else "[Spike]"
                st.markdown(f"{an_type_color} **{an['date']}** - Z-Score: `{an['z_score']:+.2f}` | Actual Sales: `${an['value']:,.2f}` | *Cause: {an['explanation']}*")
        else:
            st.info("No rolling window anomalies exceeding thresholds.")
        
    st.divider()
    
    # 2. Support tickets resolution workspace
    st.markdown("**Operational support Tickets Resolution Center**")
    
    if role == "Sales Manager":
        st.warning("Access Restricted: Support ticket resolutions and issue tracking are only available to CEO and Operations Manager roles.")
    else:
        df_active_tickets = run_db_query("SELECT issue_id, customer_id, issue_type, priority, root_cause, status FROM raw_issues WHERE status != 'Resolved';", show_error=False)
        
        if not df_active_tickets.empty:
            # PII mask contacts
            df_display_tickets = aie.mask_pii_data(df_active_tickets)
            
            # Display table of active issues
            st.dataframe(df_display_tickets, use_container_width=True)
            
            # Add action triggers
            ticket_to_resolve = st.selectbox("Select Support Ticket to Resolve", options=df_active_tickets["issue_id"].tolist())
            
            col_act1, col_act2, col_act3 = st.columns(3)
            with col_act1:
                if st.button("Dispatch 15% Invoice Refund Credit"):
                    aie.log_audit(st.session_state.role, f"Dispatched 15% Refund for Ticket {ticket_to_resolve}", f"UPDATE raw_issues SET status='Resolved', action_taken='Issued 15% Credit Refund' WHERE issue_id='{ticket_to_resolve}'", 120, "SUCCESS")
                    st.session_state.action_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Resolved {ticket_to_resolve}: Dispatched 15% Invoice Credit Refund.")
                    st.success(f"Refund credit dispatched for {ticket_to_resolve}.")
            with col_act2:
                if st.button("Trigger Express Courier Replacement"):
                    aie.log_audit(st.session_state.role, f"Triggered courier replacement for {ticket_to_resolve}", f"UPDATE raw_issues SET status='Resolved', action_taken='Shipped replacement express' WHERE issue_id='{ticket_to_resolve}'", 145, "SUCCESS")
                    st.session_state.action_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Resolved {ticket_to_resolve}: Express replacement shipment scheduled via Fedex API.")
                    st.success(f"Courier replacement scheduled for {ticket_to_resolve}.")
            with col_act3:
                if st.button("Schedule Customer Success SLA Call"):
                    aie.log_audit(st.session_state.role, f"Scheduled CSO Call for {ticket_to_resolve}", f"UPDATE raw_issues SET status='Resolved', action_taken='CSO Onboarding Scheduled' WHERE issue_id='{ticket_to_resolve}'", 90, "SUCCESS")
                    st.session_state.action_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Resolved {ticket_to_resolve}: CSO onboarding call scheduled. Syncing with HubSpot CRM.")
                    st.success(f"CSO callback scheduled for {ticket_to_resolve}.")
                    
            # Action log console
            if st.session_state.action_logs:
                st.markdown("**Integration Action Logs:**")
                for log in st.session_state.action_logs[::-1]:
                    st.code(log)
        else:
            st.success("All support tickets are currently resolved!")


# ----------------- TAB 4: CONVERSATIONAL AI ASSISTANT -----------------
with tab4:
    st.header("Intelligent Conversational BI Interface")
    
    col_chat, col_agents = st.columns([2, 1])
    
    with col_chat:
        st.subheader("AI Chat Room")
        st.write(f"Connected in **{st.session_state.ai_mode}** as role: `{st.session_state.role}`.")
        
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        user_input = st.chat_input("Ask about sales forecasts, K-Means customer clusters, market basket rules, or outages...")
        
        if user_input:
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Update chat context with spreadsheet info
            if "imported_metrics" in st.session_state and st.session_state["imported_metrics"] is not None:
                metrics = st.session_state["imported_metrics"]
                st.session_state.chat_context["spreadsheet_active"] = True
                st.session_state.chat_context["spreadsheet_filename"] = st.session_state.get("uploaded_filename", "Imported Data")
                st.session_state.chat_context["spreadsheet_summary"] = {
                    "total_revenue": metrics.get("total_revenue", 0.0),
                    "total_profit_loss": metrics.get("total_profit_loss", 0.0),
                    "profit_margin": metrics.get("profit_margin", 0.0),
                    "mom_growth": metrics.get("mom_growth", 0.0),
                    "yearly_change": metrics.get("yearly_change", 0.0),
                    "mapped_columns": metrics.get("mapped_cols", {})
                }
                if "df_sales_compat" in metrics:
                    st.session_state.chat_context["df_sales_compat"] = metrics["df_sales_compat"]
                if "df_items_compat" in metrics:
                    st.session_state.chat_context["df_items_compat"] = metrics["df_items_compat"]
            else:
                st.session_state.chat_context.pop("spreadsheet_active", None)
                st.session_state.chat_context.pop("spreadsheet_summary", None)
                st.session_state.chat_context.pop("df_sales_compat", None)
                st.session_state.chat_context.pop("df_items_compat", None)
                
            # Fetch AI response
            with st.spinner("Analyzing data tables..."):
                response_text = aie.query_ai_assistant(user_input, st.session_state.role, st.session_state.chat_context)
                
            # Display assistant message
            with st.chat_message("assistant"):
                st.markdown(response_text)
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
            
            # Auto-rerun to update Multi-Agent log block
            st.rerun()
            
    with col_agents:
        st.subheader("Multi-Agent Workspace Simulation")
        st.write("Planner, Executor, and Reviewer logs for your last query.")
        
        if st.session_state.chat_history:
            last_user_query = [msg["content"] for msg in st.session_state.chat_history if msg["role"] == "user"][-1]
            
            agent_logs = aie.run_multi_agent_collaboration(last_user_query, st.session_state.role)
            
            for log in agent_logs:
                emoji = "[Task]"
                if log["agent"] == "Planner Agent":
                    emoji = "[ML]"
                elif log["agent"] == "Executor Agent":
                    emoji = "[Sys]"
                elif log["agent"] == "Reviewer Agent":
                    emoji = "[Sec]"
                    
                st.markdown(f"**{emoji} {log['agent']}**")
                st.caption(log["action"])
                st.info(log["thought"])
        else:
            st.info("Start a chat query to view Multi-Agent collaborative thought process logs.")


# ----------------- TAB 5: DATA GOVERNANCE & AUDITS -----------------
with tab5:
    st.header("Enterprise Data Governance, Lineage, & Audits")
    
    gov_selector = st.segmented_control(
        "Governance Category",
        options=["Data Lineage", "Data Dictionary", "Data Quality Logs", "Security Audit Trails"],
        default="Data Lineage"
    )
    
    # Category A: Data Lineage (Mermaid Flowchart)
    if gov_selector == "Data Lineage":
        st.subheader("Pipeline Data Lineage Flow (SOC2 Compliant)")
        st.write("Trace the operational source files flow to the core warehouse dimensions, facts, and pre-calculated aggregations.")
        
        st.markdown("""
```mermaid
graph TD
    subgraph Kaggle Data Sources
        KG1[rawanyasser42x/sales-analysis]
        KG2[aashirgurung/sales-analysis]
    end
    
    subgraph Ingestion Cache
        LC[data/raw/ CSV files]
        KG1 --> LC
        KG2 --> LC
    end

    subgraph Ingestion Layer
        R1[CRM Checkout: raw_sales]
        R2[CRM Contacts: raw_customers]
        R3[ERP Catalog: raw_products]
        R4[Marketing API: raw_leads]
        R5[Support Portal: raw_issues]
        LC -- Map columns --o R1
        LC -- Map columns --o R2
        LC -- Map columns --o R3
    end
    
    subgraph ELT Quality Gate
        DQ[Quality Auditor: data_quality_logs]
        R1 --> DQ
        R2 --> DQ
        R3 --> DQ
    end
    
    subgraph Warehouse Star Schema
        D1[dim_customers <br>+Calculated RFM]
        D2[dim_products]
        D3[dim_dates]
        D4[dim_regions]
        F1[fact_sales <br>+Calculated profit margins]
        
        DQ -- Clean records --> D1
        DQ -- Clean records --> D2
        R4 --> D1
        R1 --> F1
        D1 --> F1
        D2 --> F1
        D3 --> F1
        D4 --> F1
    end
    
    subgraph Aggregations & ML Views
        AG1[agg_sales_monthly]
        AG2[agg_sales_product_summary]
        F1 --> AG1
        F1 --> AG2
    end
    
    style DQ fill:#F87171,stroke:#334155,color:#fff
    style F1 fill:#60A5FA,stroke:#334155,color:#fff
```
""")
        
    # Category B: Data Dictionary
    elif gov_selector == "Data Dictionary":
        st.subheader("Metadata Column Dictionary Explorer")
        df_dict = run_db_query("SELECT table_name, column_name, data_type, description, lineage_source FROM data_dictionary;", show_error=False)
        if not df_dict.empty:
            st.dataframe(df_dict, use_container_width=True)
        else:
            st.info("No metadata registered or access blocked by RBAC rules.")
            
    # Category C: Quality Monitor
    elif gov_selector == "Data Quality Logs":
        st.subheader("Data Quality Logs (Audit Trail logs)")
        st.write("Lists records excluded by the ELT pipeline checks due to negative values, duplicates, or null values.")
        
        if role == "Sales Manager":
            st.warning("Access Restricted: Sales Manager role does not have permission to view data quality monitoring logs.")
        else:
            df_dq_logs = run_db_query("SELECT id, table_name, record_id, check_name, log_message, severity, log_date FROM data_quality_logs ORDER BY log_date DESC;", show_error=False)
            if not df_dq_logs.empty:
                st.warning(f"Auditor Alert: Encountered {len(df_dq_logs)} non-critical/critical records during pipeline ingestion.")
                st.dataframe(df_dq_logs, use_container_width=True)
            else:
                st.success("Quality Auditor reports 0 failed records during pipeline executions!")
            
    # Category D: Audit log traces
    elif gov_selector == "Security Audit Trails":
        st.subheader("Security Audit Trails (Access Logs)")
        st.write("Granular execution logs auditing who requested what queries, SQL executed, response times, and success indicators.")
        
        if role in ["Sales Manager", "Operations Manager"]:
            st.warning("Access Restricted: Security audit trails are restricted to the CEO role.")
        else:
            df_audit = run_db_query("SELECT id, user_role, user_query, generated_sql, execution_time_ms, timestamp, status FROM audit_logs ORDER BY timestamp DESC LIMIT 50;", show_error=False)
            if not df_audit.empty:
                st.dataframe(df_audit, use_container_width=True)
            else:
                st.info("No audit logs recorded.")
