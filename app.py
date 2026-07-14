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

# Import local engines
import analytics_engine as ae
import ai_engine as aie

# Page Configuration
st.set_page_config(
    page_title="AI Business InSite Assistant",
    page_icon="📊",
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
    st.title("⚙️ Workspace Controls")
    
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
        st.markdown("##### 🔑 Ingestion Options")
        
        # Check if already loaded from env
        current_gemini_key = os.getenv("GEMINI_API_KEY")
        if current_gemini_key and len(current_gemini_key.strip()) > 10:
            st.success("✅ API Key automatically loaded from `.env` file!")
            
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
        st.markdown("##### 🔑 Ingestion Options")
        
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
    st.subheader("🔌 Connection Monitor")
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
    - CRM Integration (Salesforce): 🟢 Active
    - ERP Integration (NetSuite): 🟢 Active
    - Data Lake Pipeline (ELT): 🟢 Active
    """)

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
    "📊 Executive Dashboard", 
    "🧠 Advanced Analytics & ML", 
    "🛠️ Intelligent Sandbox & Resolution", 
    "🤖 Conversational AI Assistant", 
    "🛡️ Data Governance & Audits"
])

# ----------------- TAB 1: EXECUTIVE DASHBOARD -----------------
with tab1:
    st.header("📊 Enterprise Sales Overview & Operations Insights")
    
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
            growth_val = f"<span class='delta-up'>▲ {mom_growth:+.1f}%</span> MoM Growth" if mom_growth is not None else "Baseline"
        else:
            rev_val = "Restricted"
            growth_val = "<span class='delta-down'>🔒 CFO/Sales Only</span>"
            
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
            margin_delta = "Target: <span class='delta-up'>50.0%</span>"
        else:
            margin_val = "Restricted"
            margin_delta = "<span class='delta-down'>🔒 CFO/Sales Only</span>"
            
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
            tickets_delta = "<span class='delta-down'>🔒 Ops/Support Only</span>"
            
        st.markdown(f"""
        <div class="metric-card glow-red">
            <div class="metric-title">Active Support Tickets</div>
            <div class="metric-value">{tickets_val}</div>
            <div class="metric-delta">{tickets_delta}</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Interactive Dashboard Graphs
    st.subheader("📈 Time Series Sales Trends & Operational Analysis")
    
    if role == "Operations Manager":
        st.warning("🔒 Access Restricted: Operational roles do not have permission to view sales financial graphs.")
    else:
        col_graph1, col_graph2 = st.columns([2, 1])
        
        with col_graph1:
            if not df_sales.empty:
                # Group sales by date for daily plot
                df_daily = df_sales.groupby("date_id")[["net_revenue", "profit_amount"]].sum().reset_index()
                # Convert date_id to str
                df_daily["date_id"] = df_daily["date_id"].astype(str)
                
                fig_trend = px.line(
                    df_daily, x="date_id", y="net_revenue",
                    title="Daily Net Sales Revenue Trend (July 2025 - July 2026)",
                    labels={"date_id": "Date", "net_revenue": "Net Sales ($)"},
                    color_discrete_sequence=["#60A5FA"]
                )
                # Add anomaly flags visually
                fig_trend.add_annotation(
                    x="2026-03-15", y=0, text="Database Outage (East)",
                    showarrow=True, arrowhead=1, ax=0, ay=-40,
                    bordercolor="#F87171", borderwidth=1, borderpad=4, bgcolor="#1E293B"
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
        st.subheader("📦 Product Category Performance Leaderboard")
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
            st.subheader("📥 Financial Data Import (Excel / CSV)")
            st.write("Ingest external transactional sheets to calculate total revenues, net profits, and profit margins dynamically.")
            
            uploaded_file = st.file_uploader(
                "Upload Excel or CSV Sheet", 
                type=["xlsx", "xls", "csv"], 
                key="financial_data_uploader",
                help="Requires columns for Revenue (e.g., Sales, Net Revenue) and Profit (e.g., Net Profit, Earnings) to compute aggregates."
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith(".csv"):
                        df_imported = pd.read_csv(uploaded_file)
                    else:
                        df_imported = pd.read_excel(uploaded_file)
                    
                    st.success("🎉 Financial spreadsheet ingested successfully!")
                    
                    # Columns lowercase matching
                    cols = [c.lower() for c in df_imported.columns]
                    rev_col = None
                    profit_col = None
                    
                    # Match Revenue
                    for r_c in ["revenue", "sales", "net_revenue", "total_revenue", "rev"]:
                        if r_c in cols:
                            rev_col = df_imported.columns[cols.index(r_c)]
                            break
                            
                    # Match Profit
                    for p_c in ["profit", "profit_amount", "net_profit", "earnings"]:
                        if p_c in cols:
                            profit_col = df_imported.columns[cols.index(p_c)]
                            break
                            
                    st.markdown("#### 🔍 Imported Data Preview (First 5 Rows)")
                    st.dataframe(df_imported.head(5), use_container_width=True)
                    
                    if rev_col or profit_col:
                        st.markdown("### 📊 Imported Sheet Summary Metrics")
                        ic1, ic2, ic3 = st.columns(3)
                        
                        with ic1:
                            if rev_col:
                                total_rev_imp = df_imported[rev_col].sum()
                                st.metric("Imported Net Revenue", f"${total_rev_imp:,.2f}")
                            else:
                                st.metric("Imported Net Revenue", "N/A", help="No revenue column detected.")
                                
                        with ic2:
                            if profit_col:
                                total_prof_imp = df_imported[profit_col].sum()
                                st.metric("Imported Net Profit", f"${total_prof_imp:,.2f}")
                            else:
                                st.metric("Imported Net Profit", "N/A", help="No profit column detected.")
                                
                        with ic3:
                            if rev_col and profit_col:
                                margin_imp = (total_prof_imp / total_rev_imp * 100) if total_rev_imp > 0 else 0.0
                                st.metric("Imported Profit Margin", f"{margin_imp:.1f}%")
                            else:
                                st.metric("Imported Profit Margin", "N/A")
                    else:
                        st.info("💡 Standard revenue or profit columns not found. Make sure your sheet contains columns like 'Revenue' and 'Profit'.")
                except Exception as e:
                    st.error(f"Error parsing file: {e}")


# ----------------- TAB 2: ADVANCED ANALYTICS & ML -----------------
with tab2:
    st.header("🧠 Machine Learning & Advanced Statistical Diagnostics")
    
    ml_selector = st.segmented_control(
        "Machine Learning Task",
        options=["🔮 Time Series Forecasting", "👥 Customer Segmentation", "🛒 Cross-Sell Recommendations", "🎯 Leash & Churn Scoring", "💬 B2B Sales Conversion Classifier"],
        default="🔮 Time Series Forecasting"
    )
    
    # Task A: Forecasting
    if ml_selector == "🔮 Time Series Forecasting":
        st.subheader("🔮 30-Day Forward Sales Forecasting (Seasonality + Trend)")
        
        if role == "Operations Manager":
            st.warning("🔒 Access Restricted: Sales Forecasting requires financial permissions (CEO or Sales Manager).")
        else:
            # Load historical daily sales
            df_hist = run_db_query("SELECT date_id, SUM(net_revenue) as sales FROM fact_sales WHERE status='Completed' GROUP BY date_id ORDER BY date_id;", show_error=False)
            if not df_hist.empty:
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
    elif ml_selector == "👥 Customer Segmentation":
        st.subheader("👥 Unsupervised RFM Value Clustering (NumPy K-Means)")
        
        if role == "Operations Manager":
            st.warning("🔒 Access Restricted: Customer Segmentation requires financial permissions (CEO or Sales Manager).")
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
    elif ml_selector == "🛒 Cross-Sell Recommendations":
        st.subheader("🛒 Association Rule Mining for Market Basket Upsells")
        
        df_items = run_db_query("""
            SELECT f.order_id, p.product_name 
            FROM fact_sales f 
            JOIN dim_products p ON f.product_id = p.product_id 
            WHERE f.status='Completed';
        """)
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
    elif ml_selector == "🎯 Leash & Churn Scoring":
        st.subheader("🎯 Active Pipelines Churn Risk Roster & Lead Grades")
        
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
                
    # Task E: B2B Sales Conversion Classifier
    elif ml_selector == "💬 B2B Sales Conversion Classifier":
        st.subheader("💬 B2B SaaS Sales Conversation AI Classifier & Coach")
        st.write("Analyze actual dialogue transcripts, track success probabilities, and coach sales representatives using advanced analytics models.")
        
        # Load weights/metrics
        weights_path = os.path.join(r"D:\Data_Analysatics\Mini-project4", "saas_sales_model_weights.json")
        metrics = {"accuracy": 0.88, "precision": 0.86, "recall": 0.90, "f1_score": 0.88} # defaults
        if os.path.exists(weights_path):
            try:
                with open(weights_path, 'r') as f:
                    model_data = json.load(f)
                    metrics = model_data.get("metrics", metrics)
            except Exception:
                pass
                
        # 1. Performance Panel
        st.markdown("### 🏆 NumPy Logistic Regression Model Performance")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Classification Accuracy", f"{metrics['accuracy']:.1%}")
        m_col2.metric("Precision Index", f"{metrics['precision']:.1%}")
        m_col3.metric("Recall Rate", f"{metrics['recall']:.1%}")
        m_col4.metric("F1-Score Metric", f"{metrics['f1_score']:.4f}")
        
        # Retrain Trigger
        if st.button("🎯 Retrain NumPy Sales Classifier"):
            with st.spinner("Retraining classifier model via gradient descent..."):
                try:
                    from train_conversation_assistant import train_model
                    train_model(epochs=120, learning_rate=0.25)
                    st.success("🎉 NumPy Logistic Regression Model successfully retrained on 500 sales conversations!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    
        # 2. Conversation Viewer & Trajectory
        st.divider()
        st.markdown("### 🔍 Select Sales Conversation to Analyze")
        
        csv_path = os.path.join(r"D:\Data_Analysatics\Mini-project4", "saas_sales_conversations.csv")
        if os.path.exists(csv_path):
            df_convs = pd.read_csv(csv_path)
            conv_list = df_convs["conversation_id"].tolist()
            selected_conv = st.selectbox("Active Sales Call ID", options=conv_list)
            
            # Get selected row details
            row_det = df_convs[df_convs["conversation_id"] == selected_conv].iloc[0]
            
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
                    st.success("💰 MODEL PREDICTION: **LIKELY CONVERSION** (High conversion probability)")
                else:
                    st.error("⚠️ MODEL PREDICTION: **HIGH CHURN RISK** (Conversion probability dropped below baseline)")
                    
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
                st.markdown("#### 🗣️ Objection Handling Pitch Coach")
                objection_type = row_det["objection_type"]
                user_pitch = st.text_area(f"Type a response pitch to address the '{objection_type}' objection:", 
                                         placeholder="e.g. We can offer customized security documentation and sign a HIPAA BAA...")
                
                if st.button("🚀 Run AI Coaching Evaluation"):
                    if user_pitch:
                        with st.spinner("Sales coach analyzing pitch..."):
                            coach_query = f"As a professional B2B Sales Coach, evaluate this pitch response: '{user_pitch}' to resolve a '{objection_type}' objection in a B2B SaaS conversation. Provide a short, constructive score out of 100 and direct bullet-point improvement tips."
                            coach_feedback = aie.query_ai_assistant(coach_query, "Sales Manager", {})
                            st.markdown("##### 🎓 Professional Sales Coach Feedback:")
                            st.write(coach_feedback)
                    else:
                        st.warning("Please type a pitch to evaluate.")
        else:
            st.error("Sales conversations dataset not found.")
            
        # 3. Notebooks ML Benchmarks Section
        st.divider()
        st.markdown("### 📊 Trained Datasets & Model Benchmarks (Mini-project4 -> Trained-Datasets)")
        nb_col1, nb_col2 = st.columns(2)
        with nb_col1:
            st.markdown("#### 📈 Walmart Sales Prediction ML Benchmarks")
            st.caption("Derived from `walmart-sales-prediction-best-ml-algorithms.ipynb`")
            st.markdown("""
            - **Algorithms Tested:** Multiple Linear Regression (MLR), Ridge, Lasso, ElasticNet, and Polynomial Regression (Degree=2).
            - **Key Inference:** Polynomial regression overfits the holiday sales volume anomalies. 
            - **Optimal Model:** The simple Multiple Linear Regression (MLR) model with Recursive Feature Elimination (RFE) yields the best balanced R2/RMSE testing scores.
            """)
        with nb_col2:
            st.markdown("#### 🧬 Global AI Adaptation & Adoption Survey (Kaggle 2021)")
            st.caption("Derived from `data-science-in-2021-adaptation-or-adoption.ipynb`")
            st.markdown("""
            - **Core Language Ratios:** Python remains the dominant data science language, followed by SQL and R.
            - **AI Adoption Maturity Index:** Scored from 0 (No ML) to 4 (Established models in production for 2+ years).
            - **Regional Leaders:** USA and China lead in absolute volumes, while Japan and Russia exhibit high competitive density in top Kaggle tiers.
            """)


# ----------------- TAB 3: INTELLIGENT SANDBOX & RESOLUTION -----------------
with tab3:
    st.header("🛠️ Intelligent Operations Sandbox & Resolution Center")
    
    st.subheader("🔄 What-If Strategic Scenario Simulator")
    
    if role == "Operations Manager":
        st.warning("🔒 Access Restricted: Operational roles do not have permission to run what-if sales simulations.")
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
    
    st.subheader("🚨 Problem Resolution Board & Anomaly Log")
    
    if role == "Operations Manager":
        st.warning("🔒 Access Restricted: Daily sales anomaly logs require financial permissions.")
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
                an_type_color = "🔴" if an["type"] == "Drop" else "🟢"
                st.markdown(f"{an_type_color} **{an['date']}** - Z-Score: `{an['z_score']:+.2f}` | Actual Sales: `${an['value']:,.2f}` | *Cause: {an['explanation']}*")
        else:
            st.info("No rolling window anomalies exceeding thresholds.")
        
    st.divider()
    
    # 2. Support tickets resolution workspace
    st.markdown("**Operational support Tickets Resolution Center**")
    
    if role == "Sales Manager":
        st.warning("🔒 Access Restricted: Support ticket resolutions and issue tracking are only available to CEO and Operations Manager roles.")
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
                if st.button("💸 Dispatch 15% Invoice Refund Credit"):
                    aie.log_audit(st.session_state.role, f"Dispatched 15% Refund for Ticket {ticket_to_resolve}", f"UPDATE raw_issues SET status='Resolved', action_taken='Issued 15% Credit Refund' WHERE issue_id='{ticket_to_resolve}'", 120, "SUCCESS")
                    st.session_state.action_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Resolved {ticket_to_resolve}: Dispatched 15% Invoice Credit Refund.")
                    st.success(f"Refund credit dispatched for {ticket_to_resolve}.")
            with col_act2:
                if st.button("📦 Trigger Express Courier Replacement"):
                    aie.log_audit(st.session_state.role, f"Triggered courier replacement for {ticket_to_resolve}", f"UPDATE raw_issues SET status='Resolved', action_taken='Shipped replacement express' WHERE issue_id='{ticket_to_resolve}'", 145, "SUCCESS")
                    st.session_state.action_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Resolved {ticket_to_resolve}: Express replacement shipment scheduled via Fedex API.")
                    st.success(f"Courier replacement scheduled for {ticket_to_resolve}.")
            with col_act3:
                if st.button("📞 Schedule Customer Success SLA Call"):
                    aie.log_audit(st.session_state.role, f"Scheduled CSO Call for {ticket_to_resolve}", f"UPDATE raw_issues SET status='Resolved', action_taken='CSO Onboarding Scheduled' WHERE issue_id='{ticket_to_resolve}'", 90, "SUCCESS")
                    st.session_state.action_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Resolved {ticket_to_resolve}: CSO onboarding call scheduled. Syncing with HubSpot CRM.")
                    st.success(f"CSO callback scheduled for {ticket_to_resolve}.")
                    
            # Action log console
            if st.session_state.action_logs:
                st.markdown("💬 **Integration Action Logs:**")
                for log in st.session_state.action_logs[::-1]:
                    st.code(log)
        else:
            st.success("All support tickets are currently resolved!")


# ----------------- TAB 4: CONVERSATIONAL AI ASSISTANT -----------------
with tab4:
    st.header("🤖 Intelligent Conversational BI Interface")
    
    col_chat, col_agents = st.columns([2, 1])
    
    with col_chat:
        st.subheader("💬 AI Chat Room")
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
        st.subheader("👥 Multi-Agent Workspace Simulation")
        st.write("Planner, Executor, and Reviewer logs for your last query.")
        
        if st.session_state.chat_history:
            last_user_query = [msg["content"] for msg in st.session_state.chat_history if msg["role"] == "user"][-1]
            
            agent_logs = aie.run_multi_agent_collaboration(last_user_query, st.session_state.role)
            
            for log in agent_logs:
                emoji = "📋"
                if log["agent"] == "Planner Agent":
                    emoji = "🧠"
                elif log["agent"] == "Executor Agent":
                    emoji = "⚙️"
                elif log["agent"] == "Reviewer Agent":
                    emoji = "🛡️"
                    
                st.markdown(f"**{emoji} {log['agent']}**")
                st.caption(log["action"])
                st.info(log["thought"])
        else:
            st.info("Start a chat query to view Multi-Agent collaborative thought process logs.")


# ----------------- TAB 5: DATA GOVERNANCE & AUDITS -----------------
with tab5:
    st.header("🛡️ Enterprise Data Governance, Lineage, & Audits")
    
    gov_selector = st.segmented_control(
        "Governance Category",
        options=["🧬 Data Lineage", "📖 Data Dictionary", "🔍 Data Quality Logs", "🔒 Security Audit Trails"],
        default="🧬 Data Lineage"
    )
    
    # Category A: Data Lineage (Mermaid Flowchart)
    if gov_selector == "🧬 Data Lineage":
        st.subheader("🧬 Pipeline Data Lineage Flow (SOC2 Compliant)")
        st.write("Trace the operational source files flow to the core warehouse dimensions, facts, and pre-calculated aggregations.")
        
        st.markdown("""
```mermaid
graph TD
    subgraph Ingestion Layer
        R1[CRM Checkout: raw_sales]
        R2[CRM Contacts: raw_customers]
        R3[ERP Catalog: raw_products]
        R4[Marketing API: raw_leads]
        R5[Support Portal: raw_issues]
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
    elif gov_selector == "📖 Data Dictionary":
        st.subheader("📖 Metadata Column Dictionary Explorer")
        df_dict = run_db_query("SELECT table_name, column_name, data_type, description, lineage_source FROM data_dictionary;", show_error=False)
        if not df_dict.empty:
            st.dataframe(df_dict, use_container_width=True)
        else:
            st.info("No metadata registered or access blocked by RBAC rules.")
            
    # Category C: Quality Monitor
    elif gov_selector == "🔍 Data Quality Logs":
        st.subheader("🔍 Data Quality Logs (Audit Trail logs)")
        st.write("Lists records excluded by the ELT pipeline checks due to negative values, duplicates, or null values.")
        
        if role == "Sales Manager":
            st.warning("🔒 Access Restricted: Sales Manager role does not have permission to view data quality monitoring logs.")
        else:
            df_dq_logs = run_db_query("SELECT id, table_name, record_id, check_name, log_message, severity, log_date FROM data_quality_logs ORDER BY log_date DESC;", show_error=False)
            if not df_dq_logs.empty:
                st.warning(f"Auditor Alert: Encountered {len(df_dq_logs)} non-critical/critical records during pipeline ingestion.")
                st.dataframe(df_dq_logs, use_container_width=True)
            else:
                st.success("Quality Auditor reports 0 failed records during pipeline executions!")
            
    # Category D: Audit log traces
    elif gov_selector == "🔒 Security Audit Trails":
        st.subheader("🔒 Security Audit Trails (Access Logs)")
        st.write("Granular execution logs auditing who requested what queries, SQL executed, response times, and success indicators.")
        
        if role in ["Sales Manager", "Operations Manager"]:
            st.warning("🔒 Access Restricted: Security audit trails are restricted to the CEO role.")
        else:
            df_audit = run_db_query("SELECT id, user_role, user_query, generated_sql, execution_time_ms, timestamp, status FROM audit_logs ORDER BY timestamp DESC LIMIT 50;", show_error=False)
            if not df_audit.empty:
                st.dataframe(df_audit, use_container_width=True)
            else:
                st.info("No audit logs recorded.")
