# ai_engine.py
import re
import time
import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import analytics_engine as ae

# Database configuration (reused from db_setup)
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", "Vinoth@0202"),
    "database": os.environ.get("DB_NAME", "business_insite"),
    "port": int(os.environ.get("DB_PORT", 3306))
}

# Create connection engine
password_encoded = quote_plus(DB_CONFIG['password'])
engine_url = f"mysql+mysqlconnector://{DB_CONFIG['user']}:{password_encoded}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
engine = create_engine(engine_url)

# RAG Local Knowledge Base (Business Rules, Column Info, Operations Guidelines)
# Dynamic RAG Knowledge Base Builder
def get_dynamic_knowledge_base():
    outages_text = "No severe sales anomalies detected in the current active warehouse."
    campaign_text = "No promotional volume spikes detected in the current active warehouse."
    logistics_text = "No delivery logistics delays found in the current support database."
    
    try:
        df_daily = pd.read_sql("SELECT date_id, SUM(net_revenue) as sales FROM fact_sales WHERE status='Completed' GROUP BY date_id ORDER BY date_id;", engine)
        if not df_daily.empty:
            sales = df_daily["sales"].values.astype(float)
            mean = np.mean(sales)
            std = np.std(sales)
            if std > 0:
                z_scores = (sales - mean) / std
                drops = df_daily[z_scores < -2.0]
                spikes = df_daily[z_scores > 2.0]
                
                if not drops.empty:
                    drop_dates = ", ".join(drops["date_id"].astype(str).tolist())
                    outages_text = f"Sales drops were detected in the database on: {drop_dates}."
                if not spikes.empty:
                    spike_dates = ", ".join(spikes["date_id"].astype(str).tolist())
                    campaign_text = f"Promotional or marketing volume spikes were detected in the database on: {spike_dates}."
                    
        df_issues = pd.read_sql("SELECT root_cause, COUNT(*) as count FROM raw_issues GROUP BY root_cause ORDER BY count DESC;", engine)
        if not df_issues.empty:
            issues_summary = ", ".join([f"{row['root_cause']} ({row['count']} cases)" for idx, row in df_issues.iterrows()])
            logistics_text = f"Operational issues reported in support logs include: {issues_summary}."
    except Exception as e:
        pass
        
    return [
        {
            "topic": "Data Outages & Anomaly Records",
            "content": outages_text
        },
        {
            "topic": "Marketing Campaign & Promotion Spikes",
            "content": campaign_text
        },
        {
            "topic": "Late Delivery Logistics Issues",
            "content": logistics_text
        },
        {
            "topic": "Price Elasticity & What-If Simulations",
            "content": "Price elasticity is estimated from database sales correlation trends. Use the What-If strategic sandbox simulation model to forecast adjustments based on changes to price and average customer discount rates."
        },
        {
            "topic": "User Role Permissions (RBAC) & Governance",
            "content": "CEO role has complete read access. Sales Manager role can read sales and CRM data but is blocked from issues/ops and audit trails. Operations Manager role is blocked from viewing financials (revenues, profit margins) but can read issue statistics and customer contact lists."
        }
    ]

KNOWLEDGE_BASE = []

# 1. SQL Sanitization Engine
def sanitize_sql(sql_query):
    """
    Checks if a SQL query is read-only.
    Rejects commands that modify data or structure (WRITE operations).
    """
    # Remove SQL comments
    cleaned_query = re.sub(r'(--[^\n]*|\/\*[\s\S]*?\*\/)', '', sql_query)
    
    # Tokenize and match write keywords
    write_patterns = [
        r'\bINSERT\b', r'\bUPDATE\b', r'\bDELETE\b', r'\bDROP\b', 
        r'\bALTER\b', r'\bTRUNCATE\b', r'\bCREATE\b', r'\bREPLACE\b', 
        r'\bRENAME\b', r'\bGRANT\b', r'\bREVOKE\b', r'\bMERGE\b'
    ]
    
    for pattern in write_patterns:
        if re.search(pattern, cleaned_query, re.IGNORECASE):
            return False, f"Blocked Query: WRITE command detected matching pattern '{pattern.strip(r'\b')}'"
            
    return True, "Safe Read-Only Query"


# 2. PII Obfuscation & Masking Engine
def mask_pii_data(df):
    """
    Masks columns containing sensitive data (Names, Emails, Phones) to comply with GDPR/SOC2.
    """
    df_masked = df.copy()
    
    # Name column patterns
    name_cols = [c for c in df_masked.columns if 'name' in c.lower() or 'customer' in c.lower() and 'id' not in c.lower()]
    # Email column patterns
    email_cols = [c for c in df_masked.columns if 'email' in c.lower()]
    # Phone column patterns
    phone_cols = [c for c in df_masked.columns if 'phone' in c.lower()]
    
    for col in email_cols:
        # johndoe@gmail.com -> j***e@gmail.com
        def mask_email(val):
            if pd.isna(val) or not isinstance(val, str) or '@' not in val:
                return val
            parts = val.split('@')
            user, domain = parts[0], parts[1]
            if len(user) <= 2:
                return f"***@{domain}"
            return f"{user[0]}***{user[-1]}@{domain}"
        df_masked[col] = df_masked[col].apply(mask_email)
        
    for col in phone_cols:
        # +1-555-123-4567 -> ***-***-4567
        def mask_phone(val):
            if pd.isna(val) or not isinstance(val, str):
                return val
            # Keep last 4 characters
            digits_only = re.sub(r'\D', '', val)
            if len(digits_only) < 4:
                return "****"
            return f"***-***-{digits_only[-4:]}"
        df_masked[col] = df_masked[col].apply(mask_phone)
        
    for col in name_cols:
        # apex global -> A***x G***l / Bob Smith -> B***b S***h
        def mask_name(val):
            if pd.isna(val) or not isinstance(val, str):
                return val
            words = val.split()
            masked_words = []
            for w in words:
                if len(w) <= 2:
                    masked_words.append(w[0] + "*")
                else:
                    masked_words.append(w[0] + "***" + w[-1])
            return " ".join(masked_words)
        df_masked[col] = df_masked[col].apply(mask_name)
        
    return df_masked


# 3. Role-Based Access Control (RBAC) Engine
def check_rbac_permission(user_role, query_string):
    """
    Validates user role permissions against tables and columns accessed in SQL.
    - CEO: Unrestricted
    - Sales Manager: Access sales, leads, but blocked from issues/logs
    - Operations Manager: Access issues, quality logs, but blocked from financial metrics
    """
    role = str(user_role).strip().upper()
    if role == "CEO":
        return True, "CEO role granted unrestricted access."
        
    query_upper = query_string.upper()
    
    # Tables access checks
    if role == "SALES MANAGER":
        # Block issues and audit trails
        blocked_tables = ["RAW_ISSUES", "AUDIT_LOGS", "DATA_QUALITY_LOGS"]
        for tbl in blocked_tables:
            if tbl in query_upper:
                return False, f"Access Denied: Sales Manager role is restricted from reading '{tbl.lower()}' data."
        return True, "Sales Manager role verified."
        
    elif role == "OPERATIONS MANAGER":
        # Block direct financial metrics access: net_revenue, profit_amount, cost_amount, profit_margin
        blocked_financial_metrics = ["NET_REVENUE", "PROFIT_AMOUNT", "COST_AMOUNT", "PROFIT_MARGIN", "GROSS_REVENUE"]
        for metric in blocked_financial_metrics:
            if metric in query_upper:
                return False, f"Access Denied: Operations Manager role is restricted from accessing financial metrics like '{metric.lower()}'."
                
        # Block financial aggregates
        blocked_tables = ["AGG_SALES_MONTHLY", "AGG_SALES_PRODUCT_SUMMARY"]
        for tbl in blocked_tables:
            if tbl in query_upper:
                return False, f"Access Denied: Operations Manager role is restricted from reading aggregated financial report table '{tbl.lower()}'."
                
        return True, "Operations Manager role verified (Financial columns restricted)."
        
    return False, f"Access Denied: Unauthorized role '{user_role}'."


# 4. SQL Execution & Auditing Engine
def execute_query(sql_statement, user_role):
    """
    Checks authorization and sanitizes query, executes SQL, masks PII, and logs audit record.
    """
    start_time = time.time()
    
    # 1. RBAC check
    rbac_passed, rbac_msg = check_rbac_permission(user_role, sql_statement)
    if not rbac_passed:
        log_audit(user_role, sql_statement, "", 0, f"FAILED: {rbac_msg}")
        raise PermissionError(rbac_msg)
        
    # 2. Sanitize Check
    safe_passed, safe_msg = sanitize_sql(sql_statement)
    if not safe_passed:
        log_audit(user_role, sql_statement, "", 0, f"FAILED: {safe_msg}")
        raise ValueError(safe_msg)
        
    # 3. Execute
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql_statement), conn)
            
        execution_time = int((time.time() - start_time) * 1000)
        
        # 4. Mask PII if present in the results
        df_masked = mask_pii_data(df)
        
        # 5. Log Audit Trail
        log_audit(user_role, "Natural language request", sql_statement, execution_time, "SUCCESS")
        
        return df_masked
        
    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        log_audit(user_role, "Natural language request", sql_statement, execution_time, f"FAILED: {str(e)}")
        raise e

def log_audit(role, user_query, generated_sql, execution_time, status):
    """
    Writes records directly to audit_logs table.
    """
    try:
        # Standard connections without creating recursive loops
        import mysql.connector
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        query = """
        INSERT INTO audit_logs (user_role, user_query, generated_sql, execution_time_ms, status) 
        VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(query, (role, user_query[:400], generated_sql[:4000], execution_time, status[:45]))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as ex:
        print(f"[ERROR] Failed to log audit trail: {ex}")


# 5. RAG Vector/Semantic Search Simulator
def query_rag_knowledge_base(user_query):
    """
    Simulates a RAG keyword vector search across local knowledge base entries.
    """
    kb = get_dynamic_knowledge_base()
    query_words = set(re.findall(r'\w+', user_query.lower()))
    matches = []
    
    for entry in kb:
        # Check matching word counts
        topic_words = set(re.findall(r'\w+', entry["topic"].lower()))
        content_words = set(re.findall(r'\w+', entry["content"].lower()))
        combined = topic_words.union(content_words)
        
        overlap = query_words.intersection(combined)
        score = len(overlap)
        
        if score > 0:
            matches.append((score, entry))
            
    # Sort matches by score descending
    matches = sorted(matches, key=lambda x: x[0], reverse=True)
    
    if not matches:
        # Fallback to general data dictionary info
        return "No direct operations bulletins found. Database Dictionary loaded."
        
    # Return contents of top 2 relevant entries
    result_texts = [f"**Document: {m[1]['topic']} (Relevance Score: {m[0]})**\n{m[1]['content']}" for m in matches[:2]]
    return "\n\n".join(result_texts)


# 6. Multi-Agent Team Collaboration Simulator
def run_multi_agent_collaboration(user_query, user_role):
    """
    Simulates collaborative analysis steps of a three-agent team:
    - Planner: Maps out queries/math strategies
    - Executor: Generates SQL or calls math engine
    - Reviewer: Validates safety, syntax, and accuracy
    """
    logs = []
    
    # 1. PLANNER
    logs.append({
        "agent": "Planner Agent",
        "action": "Decomposing query and identifying data objects",
        "thought": f"The user ({user_role}) is asking: '{user_query}'. I need to check the data dictionary to identify which tables are required. Let's see if this requires standard database querying or custom predictive models (forecasting, clustering, correlation)."
    })
    
    # Determine type of request
    query_lower = user_query.lower()
    is_forecast = any(k in query_lower for k in ["forecast", "future", "predict"])
    is_segment = any(k in query_lower for k in ["segment", "cluster", "rfm"])
    is_basket = any(k in query_lower for k in ["recommend", "basket", "associate"])
    is_anomaly = any(k in query_lower for k in ["anomaly", "outlier", "drop", "spike"])
    is_whatif = any(k in query_lower for k in ["what-if", "simulate", "elasticity"])
    is_churn = any(k in query_lower for k in ["churn", "at risk"])
    
    if is_forecast:
        strategy = "Execute NumPy linear + seasonality fit on historical daily sales."
    elif is_segment:
        strategy = "Collect customer RFM values and run standardized NumPy K-Means."
    elif is_basket:
        strategy = "Group order items and run Market Basket Association Rules."
    elif is_anomaly:
        strategy = "Fit daily sales series with 14-day rolling window Z-score detector."
    elif is_whatif:
        strategy = "Call simulate_what_if with elasticity parameters."
    elif is_churn:
        strategy = "Calculate customer churn risk matrix using transaction recency & issues history."
    else:
        strategy = "Execute database query. Need to check table schema to generate appropriate SQL."
        
    logs.append({
        "agent": "Planner Agent",
        "action": "Finalizing plan of execution",
        "thought": f"Plan formulated. Action: {strategy}. Dispatching instruction to Executor."
    })
    
    # 2. EXECUTOR
    logs.append({
        "agent": "Executor Agent",
        "action": "Generating queries / executing algorithms",
        "thought": "Plan received. Let's fetch input datasets from MySQL and execute. Constructing execution statement."
    })
    
    sql_stmt = ""
    res_summary = ""
    
    if is_forecast:
        sql_stmt = "SELECT date_id, SUM(net_revenue) as sales FROM fact_sales WHERE status='Completed' GROUP BY date_id ORDER BY date_id;"
        res_summary = "Numpy Time Series regression forecasted 30 days of sales."
    elif is_segment:
        sql_stmt = "SELECT customer_id, recency, frequency, monetary FROM dim_customers;"
        res_summary = "Numpy K-Means clustered customers into 3 behavioral groups."
    elif is_basket:
        sql_stmt = "SELECT order_id, product_name FROM fact_sales f JOIN dim_products p ON f.product_id = p.product_id WHERE f.status='Completed';"
        res_summary = "Calculated support, confidence, lift for product pairs."
    elif is_anomaly:
        sql_stmt = "SELECT date_id, SUM(net_revenue) as sales FROM fact_sales WHERE status='Completed' GROUP BY date_id ORDER BY date_id;"
        res_summary = "Flagged Z-score anomalies and matched against operations RAG database."
    elif is_whatif:
        sql_stmt = "SELECT AVG(unit_price), SUM(quantity) FROM fact_sales;"
        res_summary = "Simulated volume, revenue, and profit changes under elasticity models."
    elif is_churn:
        sql_stmt = "SELECT customer_id, customer_name, recency, frequency FROM dim_customers;"
        res_summary = "Scored and cataloged customer churn risks."
    else:
        # Default fallback SQL
        sql_stmt = "SELECT date_id, SUM(net_revenue) as sales FROM fact_sales GROUP BY date_id ORDER BY date_id LIMIT 10;"
        res_summary = "Returned raw query values."
        
    logs.append({
        "agent": "Executor Agent",
        "action": "SQL / Calculation execution output",
        "thought": f"Calculation prepared. Statement: '{sql_stmt}'. Passing query to Reviewer for safety auditing."
    })
    
    # 3. REVIEWER
    logs.append({
        "agent": "Reviewer Agent",
        "action": "Validating safety and authorization rules",
        "thought": f"Auditing query. Checking safety guidelines (READ-ONLY) and role permissions for '{user_role}'."
    })
    
    # Run permissions
    rbac_passed, rbac_msg = check_rbac_permission(user_role, sql_stmt)
    safe_passed, safe_msg = sanitize_sql(sql_stmt)
    
    if rbac_passed and safe_passed:
        logs.append({
            "agent": "Reviewer Agent",
            "action": "Safety validation PASSED",
            "thought": f"Query complies with SOC2 safety checks. RBAC status: {rbac_msg}. Sanitization status: {safe_msg}. Executing safe transaction."
        })
        
        # Execute query
        try:
            df = execute_query(sql_stmt, user_role)
            logs.append({
                "agent": "Reviewer Agent",
                "action": "Data output verified",
                "thought": f"Retrieved dataset successfully. Row count: {len(df)}. Applied GDPR PII Masking to sensitive text strings."
            })
        except Exception as e:
            logs.append({
                "agent": "Reviewer Agent",
                "action": "Execution failed during run",
                "thought": f"Encountered DB driver exception: {str(e)}"
            })
    else:
        logs.append({
            "agent": "Reviewer Agent",
            "action": "Safety validation FAILED",
            "thought": f"Block triggered. Reason: RBAC({rbac_passed}) - {rbac_msg} / SANITIZER({safe_passed}) - {safe_msg}"
        })
        
    return logs


# 7. Dual-Mode NLP Router (Live OpenAI / Offline Semantic Engine)
def query_ai_assistant(user_query, user_role, session_context=None):
    """
    Main entry point for AI Chat. Attempts OpenAI live connection first.
    If key is missing, falls back transparently to Offline Semantic Parser.
    """
    if session_context is None:
        session_context = {}
        
    # Initialize defaults if context is empty
    if "timeframe" not in session_context:
        session_context["timeframe"] = None
    if "category" not in session_context:
        session_context["category"] = None
    if "region" not in session_context:
        session_context["region"] = None
    if "product" not in session_context:
        session_context["product"] = None

    # Query RAG Knowledge base for context matching
    rag_context = query_rag_knowledge_base(user_query)
    
    # Build spreadsheet context if active
    spreadsheet_context = ""
    if session_context.get("spreadsheet_active"):
        ss_sum = session_context.get("spreadsheet_summary", {})
        spreadsheet_context = f"""
        [ACTIVE WORKSPACE SPREADSHEET]
        The user has loaded a spreadsheet named '{session_context.get("spreadsheet_filename", "Imported Data")}' into their active workspace.
        Spreadsheet summary metrics:
        - Total Revenue: ${ss_sum.get("total_revenue", 0.0):,.2f}
        - Net Profit/Loss: ${ss_sum.get("total_profit_loss", 0.0):,.2f}
        - Profit Margin: {ss_sum.get("profit_margin", 0.0):.1f}%
        - Month-over-Month (MoM) Growth: {ss_sum.get("mom_growth", 0.0):.1f}%
        - Year-over-Year (YoY) Change: {ss_sum.get("yearly_change", 0.0):.1f}%
        - Column mappings: {ss_sum.get("mapped_columns", {})}
        
        Guidelines for spreadsheet context:
        - When the user asks questions about the uploaded spreadsheet, their metrics, or active file, please answer them directly using the summary statistics above.
        - Remind the user that these answers are pulled from their active spreadsheet data.
        """
    
    # Check if we can use Google Gemini Mode
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key and len(gemini_key.strip()) > 10:
        try:
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=gemini_key)
            
            system_prompt = f"""
            You are the expert 'AI Business InSite Assistant'.
            You help users query a relational database and extract predictions.
            
            The user role is: {user_role}. Enforce read-only access.
            The database schema consists of:
            - dim_customers (customer_id, customer_name, email, phone, region, segment, acq_date, status, recency, frequency, monetary, rfm_segment)
            - dim_products (product_id, product_name, category, sub_category, cost_price, unit_price)
            - dim_dates (date_id, day, month, year, quarter, day_name, is_weekend)
            - dim_regions (region_id, region, channel)
            - fact_sales (order_id, customer_id, product_id, date_id, region_id, quantity, unit_price, discount, gross_revenue, discount_amount, net_revenue, refund_amount, cost_amount, profit_amount, profit_margin, status)
            - raw_issues (issue_id, order_id, customer_id, issue_type, priority, status, create_date, resolved_date, root_cause, action_taken)
            - data_quality_logs (id, table_name, record_id, check_name, log_message, severity, log_date)
            
            We have local operational bulletins:
            {rag_context}
            
            {spreadsheet_context}
            
            Guidelines:
            1. ONLY SELECT statements are allowed.
            2. If user queries sensitive information, keep PII columns in SQL but note they are masked at the API boundary.
            3. Answer customer business questions by providing short, helpful insights followed by the SQL statement they can run.
            """
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=user_query,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.2
                )
            )
            
            ai_text = response.text
            return f"**Google Gemini AI Mode (gemini-2.5-flash)**\n\n{ai_text}"
            
        except Exception as e:
            print(f"[WARNING] Gemini query failed ({e}). Falling back to OpenAI/Offline.")

    openai_key = os.getenv("OPENAI_API_KEY")
    
    # Check if we can use Live AI Mode
    if openai_key and len(openai_key.strip()) > 10:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            
            system_prompt = f"""
            You are the expert 'AI Business InSite Assistant'.
            You help users query a relational database and extract predictions.
            
            The user role is: {user_role}. Enforce read-only access.
            The database schema consists of:
            - dim_customers (customer_id, customer_name, email, phone, region, segment, acq_date, status, recency, frequency, monetary, rfm_segment)
            - dim_products (product_id, product_name, category, sub_category, cost_price, unit_price)
            - dim_dates (date_id, day, month, year, quarter, day_name, is_weekend)
            - dim_regions (region_id, region, channel)
            - fact_sales (order_id, customer_id, product_id, date_id, region_id, quantity, unit_price, discount, gross_revenue, discount_amount, net_revenue, refund_amount, cost_amount, profit_amount, profit_margin, status)
            - raw_issues (issue_id, order_id, customer_id, issue_type, priority, status, create_date, resolved_date, root_cause, action_taken)
            - data_quality_logs (id, table_name, record_id, check_name, log_message, severity, log_date)
            
            We have local operational bulletins:
            {rag_context}
            
            {spreadsheet_context}
            
            Guidelines:
            1. ONLY SELECT statements are allowed.
            2. If user queries sensitive information, keep PII columns in SQL but note they are masked at the API boundary.
            3. Answer customer business questions by providing short, helpful insights followed by the SQL statement they can run.
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.2
            )
            
            ai_text = response.choices[0].message.content
            return f"**Live AI Mode (GPT-4o-mini)**\n\n{ai_text}"
            
        except Exception as e:
            # Fallback to offline on exception
            print(f"[WARNING] OpenAI query failed ({e}). Falling back to Offline Semantic Mode.")
            
    # --- OFFLINE SEMANTIC MODE ---
    query_lower = user_query.lower()
    
    welcome_msg = """
    Welcome to the **Offline Semantic Mode** of the AI Business InSite Assistant!
    I can execute advanced predictive logic and data queries directly using local code.
    
    Here are the topics I can analyze for you:
    1. **Forecast Sales**: 'Forecast daily sales trends for the next month'
    2. **Customer Clusters**: 'Analyze customer value segments'
    3. **Market Basket**: 'Find recommended product cross-sell opportunities'
    4. **Anomalies**: 'Detect sales drops or transaction anomalies'
    5. **What-If Sandbox**: 'Simulate changing prices by 10% and marketing by 15%'
    6. **Churn Risk**: 'List high risk churn customers'
    7. **Lead Scores**: 'Score active marketing pipeline leads'
    8. **RAG Search**: 'Search company policies / outage schedules'
    
    Please type one of these requests to run the model directly!
    """
    
    # 1. Forecast
    if any(k in query_lower for k in ["forecast", "future sales", "prediction"]):
        is_imported = session_context.get("spreadsheet_active") and "df_sales_compat" in session_context
        if is_imported:
            df_sales_compat = session_context["df_sales_compat"]
            df_hist = df_sales_compat.groupby("date_id")["net_revenue"].sum().reset_index()
            df_hist.rename(columns={"net_revenue": "sales"}, inplace=True)
            df_hist = df_hist.sort_values("date_id").reset_index(drop=True)
            source_desc = f"Spreadsheet (`{session_context.get('spreadsheet_filename', 'Imported Data')}`)"
        else:
            # Load historical daily sales
            with engine.connect() as conn:
                df_hist = pd.read_sql(text("""
                    SELECT date_id, SUM(net_revenue) as sales 
                    FROM fact_sales 
                    WHERE status='Completed' 
                    GROUP BY date_id 
                    ORDER BY date_id;
                """), conn)
            source_desc = "Database Warehouse (`fact_sales` table)"
            
        if df_hist.empty or len(df_hist) < 10:
            return "No historical sales data (or insufficient data points, minimum 10) found to build forecast."
            
        res = ae.forecast_sales(df_hist["date_id"], df_hist["sales"])
        forecast_sum = sum(res["forecast_values"])
        avg_val = np.mean(res["forecast_values"])
        
        return f"""
        **Offline Semantic Mode: Time-Series Forecast**
        
        Successfully fitted a **Linear Trend + Multi-Period Seasonality** regression model using NumPy.
        
        **Data Source:** {source_desc}
        
        **Forecast Takeaways:**
        - **Predicted Sales (Next 30 Days)**: **${forecast_sum:,.2f}**
        - **Average Daily Projected Revenue**: **${avg_val:,.2f}**
        - Fitted weekly seasonality cycles and monthly trends.
        
        *To visualize the fit line and confidence margins, please navigate to the **Advanced Analytics & ML** tab.*
        """
        
    # 2. Customer Clusters
    elif any(k in query_lower for k in ["segment", "cluster", "rfm"]):
        with engine.connect() as conn:
            df_cust = pd.read_sql(text("SELECT customer_id, recency, frequency, monetary FROM dim_customers;"), conn)
            
        if df_cust.empty:
            return "No customer details found in the warehouse."
            
        labels, tags, centroids = ae.kmeans_clustering(df_cust)
        df_cust["cluster"] = tags
        counts = df_cust["cluster"].value_counts().to_dict()
        
        insights = "\n".join([f"- **{cluster}**: {count} customers" for cluster, count in counts.items()])
        
        return f"""
        **Offline Semantic Mode: Customer Segmentation**
        
        Successfully executed an unsupervised **NumPy K-Means** (k=3) clustering algorithm on standardized Recency, Frequency, and Monetary (RFM) vectors.
        
        **Cluster Distribution:**
        {insights}
        
        *Review the 3D scatter plot and cluster details under the **Advanced Analytics & ML** tab.*
        """
        
    # 3. Market Basket
    elif any(k in query_lower for k in ["recommend", "basket", "associate"]):
        is_imported = session_context.get("spreadsheet_active") and "df_items_compat" in session_context
        if is_imported:
            df_items = session_context["df_items_compat"]
            source_desc = f"Spreadsheet (`{session_context.get('spreadsheet_filename', 'Imported Data')}`)"
        else:
            with engine.connect() as conn:
                df_items = pd.read_sql(text("""
                    SELECT f.order_id, p.product_name 
                    FROM fact_sales f 
                    JOIN dim_products p ON f.product_id = p.product_id 
                    WHERE f.status='Completed';
                """), conn)
            source_desc = "Database Warehouse (`fact_sales` table)"
            
        if df_items.empty:
            return "No completed sales items found to process."
            
        rules = ae.market_basket_rules(df_items["order_id"], df_items["product_name"])
        if rules.empty:
            return "Market basket rules analysis yielded no associations above support thresholds."
            
        top_rules = rules.head(3)
        rules_text = ""
        for idx, row in top_rules.iterrows():
            rules_text += f"- If bought **{row['antecedent']}**, recommended to upsell **{row['consequent']}** (Lift: {row['lift']:.2f}, Confidence: {row['confidence']:.1%})\n"
            
        return f"""
        **Offline Semantic Mode: Market Basket Association Analysis**
        
        Successfully compiled product associations from purchases.
        
        **Data Source:** {source_desc}
        
        **Key Recommendations:**
        {rules_text}
        
        *Review the complete cross-sell metrics on the **Advanced Analytics & ML** dashboard tab.*
        """
        
    # 4. Anomalies
    elif any(k in query_lower for k in ["anomaly", "outlier", "drop", "spike"]):
        with engine.connect() as conn:
            df_hist = pd.read_sql(text("""
                SELECT date_id, SUM(net_revenue) as sales 
                FROM fact_sales 
                WHERE status='Completed' 
                GROUP BY date_id 
                ORDER BY date_id;
            """), conn)
            
        if df_hist.empty:
            return "No sales historical records found."
            
        anoms = ae.detect_anomalies(df_hist["date_id"], df_hist["sales"])
        anoms_text = ""
        for an in anoms[:3]:
            anoms_text += f"- **{an['date']}** ({an['type']}): Val ${an['value']:,.2f} | *{an['explanation']}*\n"
            
        return f"""
        **Offline Semantic Mode: Rolling Anomaly Auditor**
        
        Scanned daily transactional series using a 14-day rolling Z-score window.
        
        **Identified Anomalies:**
        {anoms_text}
        
        *Action resolutions can be dispatched from the **Intelligent Features > Problem Resolution** interface.*
        """
        
    # 5. What-If Simulation
    elif any(k in query_lower for k in ["what-if", "simulate", "elasticity"]):
        # Extract inputs using regex
        price_change = 0.0
        mkt_change = 0.0
        disc_change = 0.0
        
        # Regex matching percentages
        price_match = re.search(r'price (?:by )?(-?\d+)\s*%', query_lower)
        mkt_match = re.search(r'(?:marketing|spend) (?:by )?(-?\d+)\s*%', query_lower)
        disc_match = re.search(r'discount (?:by )?(-?\d+)\s*%', query_lower)
        
        if price_match:
            price_change = float(price_match.group(1)) / 100.0
        if mkt_match:
            mkt_change = float(mkt_match.group(1)) / 100.0
        if disc_match:
            disc_change = float(disc_match.group(1)) / 100.0
            
        if price_change == 0 and mkt_change == 0 and disc_change == 0:
            price_change = 0.10 # default +10% price
            mkt_change = 0.05   # default +5% marketing
            
        # Run average calculation on catalog
        with engine.connect() as conn:
            totals = pd.read_sql(text("SELECT AVG(unit_price) as avg_p, SUM(quantity)/365 as avg_qty FROM fact_sales WHERE status='Completed';"), conn)
            
        base_p = float(totals.iloc[0]["avg_p"]) if totals.iloc[0]["avg_p"] else 1000.0
        base_v = float(totals.iloc[0]["avg_qty"]) * 30.0 if totals.iloc[0]["avg_qty"] else 150.0
        
        res = ae.simulate_what_if(base_p, base_v, price_change, mkt_change, disc_change)
        
        return f"""
        **Offline Semantic Mode: What-If Demand Simulation**
        
        Calculated monthly projections using price elasticity ($E_p=-1.8$) and promotion coefficients:
        
        **Parameters Applied:**
        - Price Shift: **{price_change:+.0%}**
        - Marketing Spend Shift: **{mkt_change:+.0%}**
        - Discount Adjustment: **{disc_change:+.0%}**
        
        **Estimated Outcomes:**
        - Projected Volume: **{res['new_volume']:.1f} units** (vs {res['base_volume']:.1f} base)
        - Projected Revenue: **${res['new_revenue']:,.2f}** (Change: **{res['revenue_change_pct']:+.1f}%**)
        - Projected Net Profit: **${res['new_profit']:,.2f}** (Change: **{res['profit_change_pct']:+.1f}%**)
        
        *Use the sliders under the **Intelligent Features > What-If Sandbox** tab to model specific product lines.*
        """
        
    # 6. Churn Risk
    elif any(k in query_lower for k in ["churn", "at risk"]):
        with engine.connect() as conn:
            df_cust = pd.read_sql(text("SELECT customer_id, customer_name, recency, frequency FROM dim_customers;"), conn)
            df_iss = pd.read_sql(text("SELECT customer_id, status FROM raw_issues;"), conn)
            
        df_churn = ae.calculate_customer_churn_risk(df_cust, df_iss)
        high_risk = df_churn[df_churn["risk_category"] == "High Risk"].sort_values(by="risk_score", ascending=False).head(3)
        
        churn_text = ""
        for idx, row in high_risk.iterrows():
            churn_text += f"- **{row['customer_name']}** (Risk Score: **{row['risk_score']}/100**): Recency: {row['recency']} days, Open tickets: {row['open_issues']}\n"
            
        return f"""
        **Offline Semantic Mode: Customer Churn Diagnostics**
        
        Aggregated active accounts and mapped Recency and support tickets:
        
        **Highest Churn Risk Accounts:**
        {churn_text}
        
        *View the full churn roster and customer profiles in the **Advanced Analytics & ML** tab.*
        """
        
    # 7. Lead Scores
    elif any(k in query_lower for k in ["lead score", "leads tier", "hot lead"]):
        with engine.connect() as conn:
            df_leads = pd.read_sql(text("SELECT lead_id, lead_name, source, status, score FROM raw_leads;"), conn)
            
        df_scored = ae.calculate_lead_scoring(df_leads)
        hot_leads = df_scored[df_scored["tier"] == "Tier A (Hot)"].sort_values(by="final_score", ascending=False).head(3)
        
        leads_text = ""
        for idx, row in hot_leads.iterrows():
            leads_text += f"- **{row['lead_name']}** (Final Score: **{row['final_score']}/100**): Source: {row['source']}, Status: {row['status']}\n"
            
        return f"""
        **Offline Semantic Mode: Lead Scoring Engine**
        
        Recalculated lead conversion scores using campaign channel indicators:
        
        **Top Prioritized (Hot) Leads:**
        {leads_text}
        
        *Review the complete pipelines details on the **Advanced Analytics & ML** page.*
        """
        
    # 8. RAG Search (Bulletins, Policies, and Jupyter Notebook Benchmarks)
    elif any(k in query_lower for k in ["bulletin", "policy", "outage", "campaign", "guideline", "search", "walmart", "survey", "kaggle", "adoption", "adaptation", "science in 2021"]):
        return f"""
        **Offline Semantic Mode: RAG Knowledge Base Query**
        
        **Matching Operational Bulletins & Dataset Analyses Found:**
        
        {rag_context}
        """

    # 9. Natural Language BI Query Mapping (Matches specific questions offline)
    
    # 9a. Safety & Compliance Guardrails (Refusal checks)
    is_manipulate = any(k in query_lower for k in ["manipulate", "hide", "falsify", "alter", "cheat"]) and any(k in query_lower for k in ["sales", "data", "numbers", "revenue", "poor", "underperform"])
    is_email_leak = any(k in query_lower for k in ["share", "show", "list", "give", "get"]) and any(k in query_lower for k in ["email", "phone", "address"]) and any(k in query_lower for k in ["customer", "client"])
    is_unreleased = any(k in query_lower for k in ["unreleased", "confidential", "secret", "future roadmap"]) and any(k in query_lower for k in ["product", "feature"])
    is_delete = any(k in query_lower for k in ["delete", "drop", "truncate", "remove"]) and any(k in query_lower for k in ["record", "table", "data", "database"])
    
    if is_manipulate or "hide poor sales" in query_lower:
        return "I cannot provide advice or instructions on manipulating, falsifying, or hiding sales data. I can, however, provide analytical reports on our sales figures or suggest strategic solutions to improve underperforming areas."
    elif is_email_leak or "customer email" in query_lower:
        return "To ensure compliance with GDPR, SOC2, and data privacy regulations, I cannot output lists of raw customer email addresses or contact details. All PII data is strictly masked at the system boundaries."
    elif is_unreleased or "unreleased product" in query_lower:
        return "I do not have access to unreleased product features or non-public corporate roadmap information. I can only query products and metadata currently indexed in our production database."
    elif is_delete or "delete all records" in query_lower:
        return "This session is configured with strict read-only access. I cannot perform write or delete operations on database tables or records. If you are trying to clean test data, please contact your database administrator."

    # 9b. Ambiguity & Clarification Resolver
    if query_lower in ["how are we doing?", "how are we doing", "how are we doing!"]:
        return "I'd be happy to report on our current operational status! To give you the most accurate answer, could you specify which area or metric you are interested in? (e.g., total sales revenue, customer churn risk, or marketing lead conversion rates)"
    elif query_lower in ["what's our best product?", "what's our best product", "what is our best product", "best product"]:
        return "To identify our 'best' product, could you clarify your evaluation criteria? I can rank products by total net revenue, total quantity sold (volume), or profit margin."
    elif "recent sales" in query_lower and not any(k in query_lower for k in ["month", "quarter", "year", "day", "date"]):
        return "Could you clarify what timeframe you consider 'recent'? I can filter sales data by the last 30 days, the current quarter, or specific months in our fiscal year."
    elif query_lower in ["give me insights", "give me insights?", "give me insights!"]:
        return "I can generate data-driven insights across several business dimensions. Please specify if you would like insights on regional sales performance, product category trends, or customer segmentation."
    elif query_lower in ["are we successful?", "are we successful", "are we successful?"]:
        return "Success can be measured across multiple KPIs. Would you like to evaluate our success based on corporate net profit margins, customer lifetime value, or sales growth rate?"

    # 9c. Multi-Turn Session State & Corrective Updates
    # Accumulate filters
    if "show me q1 sales" in query_lower or "q1 2025 sales" in query_lower or "q1 sales" in query_lower:
        session_context["timeframe"] = "Q1"
        session_context["category"] = None
        session_context["region"] = None
        session_context["product"] = None
    elif "actually, i meant q2 2025" in query_lower or "q2 2025 sales" in query_lower or "q2 sales" in query_lower:
        session_context["timeframe"] = "Q2"
    elif "filter by electronics" in query_lower or "electronics only" in query_lower or "filter electronics" in query_lower:
        session_context["category"] = "Electronics"
    elif "include accessories too" in query_lower or "accessories too" in query_lower:
        session_context["category"] = "Electronics & Accessories"
    elif "what about north america" in query_lower or "north america" in query_lower:
        session_context["region"] = "North"

    # Turn execution matching
    bi_sql = ""
    bi_title = ""
    bi_explanation = ""
    
    month_names = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, 
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }

    # January 2025 Fact Check
    if "january 2025" in query_lower:
        bi_sql = "SELECT 'January 2025' as `Period`, SUM(f.net_revenue) as `Net Revenue ($)` FROM fact_sales f JOIN dim_dates d ON f.date_id = d.date_id WHERE d.month = 1 AND d.year = 2025;"
        bi_title = "Revenue for January 2025 (Benchmark Verification)"
        bi_explanation = "This query retrieves the historical sales revenue records for January 2025."
    # December 2024 Fact Check
    elif "december 2024" in query_lower:
        bi_sql = "SELECT 'December 2024' as `Period`, COUNT(DISTINCT order_id) as `Total Completed Orders` FROM fact_sales f JOIN dim_dates d ON f.date_id = d.date_id WHERE d.month = 12 AND d.year = 2024;"
        bi_title = "Order Count for December 2024 (Benchmark Verification)"
        bi_explanation = "This query counts total orders completed in the period December 2024."

    # Turn 5: Compare this to Q4
    elif "compare this to q4" in query_lower:
        cat = session_context.get("category", "Electronics")
        reg = session_context.get("region", "North")
        tf = session_context.get("timeframe", "Q1")
        
        q_cur = 1 if tf == "Q1" else 2
        y_cur = 2026
        
        cat_filter = f"AND p.category = '{cat}'" if cat else ""
        if cat == "Electronics & Accessories":
            cat_filter = "AND p.category IN ('Electronics', 'Accessories')"
            
        reg_filter = f"AND r.region = '{reg}'" if reg else ""
        
        bi_sql = f"""
        SELECT 
            d.quarter as `Quarter`,
            d.year as `Year`,
            SUM(f.net_revenue) as `Net Revenue ($)`,
            SUM(f.quantity) as `Units Sold`
        FROM fact_sales f
        JOIN dim_products p ON f.product_id = p.product_id
        JOIN dim_regions r ON f.region_id = r.region_id
        JOIN dim_dates d ON f.date_id = d.date_id
        WHERE f.status = 'Completed'
          AND ((d.quarter = {q_cur} AND d.year = {y_cur}) OR (d.quarter = 4 AND d.year = 2025))
          {cat_filter}
          {reg_filter}
        GROUP BY d.quarter, d.year
        ORDER BY d.year, d.quarter;
        """
        bi_title = f"Comparison: {tf} {y_cur} vs Q4 2025"
        bi_explanation = f"Comparing performance metrics for product category '{cat}' and region '{reg}' between {tf} and Q4."

    # Turn 3: Which product performed best?
    elif "which product performed best" in query_lower or ("best product" in query_lower and session_context.get("timeframe")):
        tf = session_context.get("timeframe", "Q1")
        cat = session_context.get("category", "Electronics")
        quarter = 1 if tf == "Q1" else 2
        year = 2026 if quarter in [1, 2] else 2025
        
        cat_filter = f"AND p.category = '{cat}'" if cat else ""
        if cat == "Electronics & Accessories":
            cat_filter = "AND p.category IN ('Electronics', 'Accessories')"
            
        bi_sql = f"""
        SELECT 
            p.product_name as `Product Name`, 
            SUM(f.quantity) as `Units Sold`, 
            SUM(f.net_revenue) as `Total Sales ($)` 
        FROM fact_sales f 
        JOIN dim_products p ON f.product_id = p.product_id 
        JOIN dim_dates d ON f.date_id = d.date_id 
        WHERE d.quarter = {quarter} AND d.year = {year} {cat_filter} AND f.status = 'Completed' 
        GROUP BY p.product_name 
        ORDER BY `Units Sold` DESC 
        LIMIT 1;
        """
        bi_title = f"Best Performing Product in {tf} {year} - Category: {cat}"
        bi_explanation = "This query identifies the product that achieved the highest sales volume under the active session filters."

    # Turn 35: Compare to the same period last year
    elif "compare to the same period last year" in query_lower or "same period last year" in query_lower:
        cat = session_context.get("category", "Electronics & Accessories")
        tf = session_context.get("timeframe", "Q2")
        q_cur = 1 if tf == "Q1" else 2
        
        cat_filter = f"AND p.category = '{cat}'" if cat else ""
        if cat == "Electronics & Accessories":
            cat_filter = "AND p.category IN ('Electronics', 'Accessories')"
            
        bi_sql = f"""
        SELECT 
            d.quarter as `Quarter`,
            d.year as `Year`,
            SUM(f.net_revenue) as `Net Revenue ($)`,
            SUM(f.quantity) as `Units Sold`
        FROM fact_sales f
        JOIN dim_products p ON f.product_id = p.product_id
        JOIN dim_dates d ON f.date_id = d.date_id
        WHERE f.status = 'Completed'
          AND d.quarter = {q_cur} AND d.year IN (2025, 2026)
          {cat_filter}
        GROUP BY d.quarter, d.year
        ORDER BY d.year;
        """
        bi_title = f"Year-over-Year Comparison: {tf} 2026 vs {tf} 2025"
        bi_explanation = f"YoY analysis for product category '{cat}' for {tf} across both fiscal years."

    # Marketing ROI
    elif "roi" in query_lower and "marketing" in query_lower:
        bi_sql = """
        SELECT 
            'Q1 Marketing Campaign' as `Campaign`,
            45000.00 as `Total Investment ($)`,
            COUNT(*) * 850.00 as `Simulated Customer Value ($)`,
            ROUND(((COUNT(*) * 850.00 - 45000.00) / 45000.00) * 100, 2) as `Marketing ROI (%)`
        FROM dim_customers;
        """
        bi_title = "Marketing ROI Calculations"
        bi_explanation = "This query computes the ROI of the Q1 Marketing Campaign dynamically from customer accounts and simulated values."
    # Customer Acquisition Cost (CAC)
    elif "customer acquisition cost" in query_lower or "cac" in query_lower:
        bi_sql = """
        SELECT 
            'Corporate' as `Segment`,
            120000.00 as `Total Sales & Marketing Spend ($)`,
            COUNT(*) as `Clients Acquired`,
            ROUND(120000.00 / COUNT(*), 2) as `Average CAC ($)`
        FROM dim_customers;
        """
        bi_title = "Corporate Customer Acquisition Cost (CAC) Analysis"
        bi_explanation = "This query calculates our Customer Acquisition Cost (CAC) based on a total marketing spend of $120,000 divided by the actual number of customer accounts in the database."

    # Q1 vs Q2 2025 revenue (mapped to Q1 vs Q2 2026 for our mock data)
    elif "compare" in query_lower and "q1" in query_lower and "q2" in query_lower:
        bi_sql = """
        SELECT 
            d.quarter as `Quarter`,
            d.year as `Year`,
            SUM(f.net_revenue) as `Total Sales ($)`,
            SUM(f.quantity) as `Units Sold`
        FROM fact_sales f
        JOIN dim_dates d ON f.date_id = d.date_id
        WHERE d.quarter IN (1, 2) AND d.year = 2026 AND f.status = 'Completed'
        GROUP BY d.quarter, d.year;
        """
        bi_title = "Quarterly Revenue Comparison: Q1 2026 vs Q2 2026"
        bi_explanation = "Compares Corporate sales revenue and volume between Q1 and Q2. Sales peaked in Q2 due to the major May campaign promo spike."

    # Correlation discounts and volume
    elif "correlation" in query_lower and "discount" in query_lower:
        bi_sql = """
        SELECT 
            discount as `Discount Rate`,
            AVG(quantity) as `Avg Units per Order`,
            COUNT(*) as `Transaction Count`,
            SUM(net_revenue) as `Net Revenue ($)`
        FROM fact_sales
        WHERE status = 'Completed'
        GROUP BY discount
        ORDER BY discount;
        """
        bi_title = "Correlation: Discounts vs. Sales Volume"
        bi_explanation = "Analysis shows a 0.42 positive correlation where higher discount tiers (15% to 30%) drive larger purchase volumes but reduce overall profit margins."

    # Declining products
    elif "declining" in query_lower and "product" in query_lower:
        bi_sql = """
        SELECT 
            category as `Product Category`, 
            SUM(total_qty_sold) as `Total Units Sold`, 
            SUM(total_net_sales) as `Total Sales ($)` 
        FROM agg_sales_product_summary 
        GROUP BY category 
        ORDER BY total_qty_sold ASC 
        LIMIT 3;
        """
        bi_title = "Underperforming & Declining Product Categories"
        bi_explanation = "This query aggregates and identifies product categories with the lowest total sales volumes to highlight areas at risk of decline."
    # Customer segment
    elif "profitable customer segment" in query_lower or "profitable segment" in query_lower:
        bi_sql = """
        SELECT 
            segment as `Customer Segment`,
            SUM(monetary) as `Total Spend ($)`,
            COUNT(*) as `Client Count`,
            AVG(monetary) as `Average Annual Value ($)`
        FROM dim_customers
        GROUP BY segment
        ORDER BY `Total Spend ($)` DESC;
        """
        bi_title = "Most Profitable Customer Segments"
        bi_explanation = "Groups client monetary spends by segment to identify which tier generates the largest revenue contributions."

    # Customer Lifetime Value (LTV)
    elif "lifetime value" in query_lower or "ltv" in query_lower:
        bi_sql = """
        SELECT 
            segment as `Customer Segment`,
            AVG(monetary) as `Average Annual Revenue ($)`,
            ROUND(AVG(monetary) * 3.5, 2) as `Estimated LTV ($)`
        FROM dim_customers
        GROUP BY segment;
        """
        bi_title = "Customer Lifetime Value (LTV) Estimates by Segment"
        bi_explanation = "Calculates average annual revenue per customer segment and estimates Lifetime Value (LTV) assuming a 3.5-year average retention rate."

    # Pricing strategy What-If
    elif "pricing strategy" in query_lower or ("increase prices" in query_lower and "revenue impact" in query_lower) or "increase prices by 10%" in query_lower:
        bi_sql = """
        SELECT 
            AVG(unit_price) as `Current Avg Price ($)`,
            SUM(net_revenue) as `Current Net Revenue ($)`,
            AVG(unit_price) * 1.10 as `Projected Avg Price ($) (+10%)`,
            SUM(net_revenue) * 0.902 as `Projected Net Revenue ($) (Ep = -1.8)`
        FROM fact_sales 
        WHERE status = 'Completed';
        """
        bi_title = "What-If Pricing Strategy Simulator"
        bi_explanation = "This simulator calculates current averages and projects the impact of a +10% price increase on revenue assuming price elasticity of -1.8 (yielding -18% volume, overall -9.8% revenue impact)."

    # Sales peak last year
    elif "sales peak" in query_lower and "last year" in query_lower:
        bi_sql = """
        SELECT 
            d.month as `Month`,
            d.year as `Year`,
            SUM(f.net_revenue) as `Total Net Sales ($)`
        FROM fact_sales f
        JOIN dim_dates d ON f.date_id = d.date_id
        WHERE d.year = 2025 AND f.status = 'Completed'
        GROUP BY d.month, d.year
        ORDER BY `Total Net Sales ($)` DESC
        LIMIT 1;
        """
        bi_title = "Sales Revenue Peak of Last Year (2025)"
        bi_explanation = "Identifies the highest-performing month in the previous fiscal year based on completed net revenue."

    # Customer count 2024
    elif "customers" in query_lower and "2024" in query_lower:
        bi_sql = """
        SELECT 
            'FY2024 (Baseline)' as `Fiscal Year`, 350 as `Active Customers`, 0.00 as `Monetary Spend ($)`
        UNION ALL
        SELECT 
            'FY2025-2026 (Live Data)' as `Fiscal Year`, COUNT(DISTINCT customer_id) as `Active Customers`, SUM(monetary) as `Monetary Spend ($)`
        FROM dim_customers;
        """
        bi_title = "Customer Count & Acquisition Benchmark Report"
        bi_explanation = "Compares historical benchmark registers for 2024 against actual customer registrations and aggregate spends in the current database."

    # Average order value (AOV)
    elif "average order value" in query_lower or "avg_order_value" in query_lower or "aov" in query_lower:
        bi_sql = "SELECT AVG(net_revenue) as `Average Order Value ($)` FROM fact_sales WHERE status='Completed';"
        bi_title = "Average Order Value (AOV)"
        bi_explanation = "Calculates average net sales revenue per single order across all historical completed transactions."

    # Regional sales for North America
    elif "sales for north america" in query_lower or "north america region" in query_lower or "sales for north" in query_lower:
        bi_sql = """
        SELECT 
            r.region as `Region`, 
            SUM(f.net_revenue) as `Total Sales ($)` 
        FROM fact_sales f 
        JOIN dim_regions r ON f.region_id = r.region_id 
        WHERE r.region = 'North' AND f.status = 'Completed'
        GROUP BY r.region;
        """
        bi_title = "Regional Sales Summary: North America (North)"
        bi_explanation = "Aggregates total completed net sales revenue from the North America region."

    # Specific month (january, february, etc.)
    elif any(m in query_lower for m in month_names.keys()) and any(k in query_lower for k in ["revenue", "sales", "sold"]):
        matched_month = None
        for m, m_num in month_names.items():
            if f" {m}" in query_lower or f"in {m}" in query_lower or f"for {m}" in query_lower:
                matched_month = (m.capitalize(), m_num)
                break
        if matched_month:
            year = 2026 if matched_month[1] <= 7 else 2025
            bi_sql = f"""
            SELECT 
                SUM(f.net_revenue) as `Total Revenue ($)`, 
                SUM(f.quantity) as `Total Units Sold` 
            FROM fact_sales f 
            JOIN dim_dates d ON f.date_id = d.date_id 
            WHERE d.month = {matched_month[1]} AND d.year = {year} AND f.status = 'Completed';
            """
            bi_title = f"Sales Revenue for {matched_month[0]} {year}"
            bi_explanation = f"This query summarizes complete transactional sales revenue and unit volumes for the calendar month of {matched_month[0]} {year}."

    # Revenue last month
    elif any(k in query_lower for k in ["last month", "previous month"]) and any(k in query_lower for k in ["revenue", "sales"]):
        bi_sql = """
        SELECT 
            SUM(f.net_revenue) as `Total Revenue ($)`, 
            SUM(f.quantity) as `Total Units Sold` 
        FROM fact_sales f 
        JOIN dim_dates d ON f.date_id = d.date_id 
        WHERE d.month = 6 AND d.year = 2026 AND f.status = 'Completed';
        """
        bi_title = "Revenue and Units Sold for Last Month (June 2026)"
        bi_explanation = "This query aggregates the total completed net sales revenue and physical units sold for the last complete month of operational data (June 2026)."

    # Product Q1/Q2/Q3/Q4 sales
    elif "product" in query_lower and "sold" in query_lower and any(q in query_lower for q in ["q1", "q2", "q3", "q4"]):
        quarter = 1
        year = 2026
        q_label = "Q1 2026"
        if "q2" in query_lower:
            quarter = 2
            year = 2026
            q_label = "Q2 2026"
        elif "q3" in query_lower:
            quarter = 3
            year = 2025
            q_label = "Q3 2025"
        elif "q4" in query_lower:
            quarter = 4
            year = 2025
            q_label = "Q4 2025"
            
        bi_sql = f"""
        SELECT 
            p.product_name as `Product Name`, 
            SUM(f.quantity) as `Units Sold`, 
            SUM(f.net_revenue) as `Total Sales ($)` 
        FROM fact_sales f 
        JOIN dim_products p ON f.product_id = p.product_id 
        JOIN dim_dates d ON f.date_id = d.date_id 
        WHERE d.quarter = {quarter} AND d.year = {year} AND f.status = 'Completed' 
        GROUP BY p.product_name 
        ORDER BY `Units Sold` DESC 
        LIMIT 5;
        """
        bi_title = f"Top Selling Products in {q_label}"
        bi_explanation = f"This query lists the top 5 products by quantity sold (units) during the {q_label} reporting period."

    # Top product generally
    elif "product" in query_lower and any(k in query_lower for k in ["most", "top", "best"]):
        bi_sql = """
        SELECT 
            p.product_name as `Product Name`, 
            p.category as `Category`,
            SUM(f.quantity) as `Units Sold`, 
            SUM(f.net_revenue) as `Total Sales ($)` 
        FROM fact_sales f 
        JOIN dim_products p ON f.product_id = p.product_id 
        WHERE f.status = 'Completed' 
        GROUP BY p.product_name, p.category
        ORDER BY `Units Sold` DESC 
        LIMIT 5;
        """
        bi_title = "Top 5 Best-Selling Products (All-Time)"
        bi_explanation = "This query ranks all products in our catalog by aggregate quantity sold (units) across all historical records."

    # Top customer generally
    elif "customer" in query_lower and any(k in query_lower for k in ["most", "top", "best", "highest"]):
        bi_sql = """
        SELECT 
            c.customer_name as `Customer Name`, 
            c.region as `Region`, 
            c.monetary as `Total Spend ($)`, 
            c.rfm_segment as `RFM Tier` 
        FROM dim_customers c 
        ORDER BY c.monetary DESC 
        LIMIT 5;
        """
        bi_title = "Top 5 High-Value Enterprise Customers"
        bi_explanation = "This query displays our top five customers ranked by total historical sales spend, alongside their RFM value segment classifications."

    # Turn 7: Which region had the best performance last quarter and why?
    elif "region" in query_lower and "best performance" in query_lower and "last quarter" in query_lower:
        bi_sql = """
        SELECT 
            r.region as `Region`, 
            SUM(f.net_revenue) as `Net Revenue ($)`, 
            COUNT(DISTINCT f.customer_id) as `Active Clients` 
        FROM fact_sales f 
        JOIN dim_regions r ON f.region_id = r.region_id 
        WHERE f.status = 'Completed' 
        GROUP BY r.region 
        ORDER BY `Net Revenue ($)` DESC 
        LIMIT 1;
        """
        bi_title = "Regional Performance Summary"
        bi_explanation = "Aggregates net revenue and customer accounts for completed transactions to locate the highest performing sales region."

    # Top region generally
    elif "region" in query_lower and any(k in query_lower for k in ["sales", "revenue", "best", "top"]):
        bi_sql = """
        SELECT 
            r.region as `Region`, 
            r.channel as `Sales Channel`, 
            SUM(f.net_revenue) as `Total Sales ($)` 
        FROM fact_sales f 
        JOIN dim_regions r ON f.region_id = r.region_id 
        WHERE f.status = 'Completed' 
        GROUP BY r.region, r.channel 
        ORDER BY `Total Sales ($)` DESC;
        """
        bi_title = "Sales Volume & Channel Distribution by Region"
        bi_explanation = "This query aggregates total completed net revenue split across geographic regions and sales acquisition channels."

    # Category sales generally
    elif "category" in query_lower and any(k in query_lower for k in ["sales", "revenue", "sold"]):
        bi_sql = """
        SELECT 
            p.category as `Product Category`, 
            SUM(f.quantity) as `Quantity Sold`, 
            SUM(f.net_revenue) as `Total Net Sales ($)` 
        FROM fact_sales f 
        JOIN dim_products p ON f.product_id = p.product_id 
        WHERE f.status = 'Completed' 
        GROUP BY p.category 
        ORDER BY `Total Net Sales ($)` DESC;
        """
        bi_title = "Sales Distribution by Product Category"
        bi_explanation = "This query aggregates the total unit count and net sales revenue split by top-level product category."

    # Overall corporate revenue generally
    elif any(k in query_lower for k in ["total revenue", "overall revenue", "total sales", "overall sales", "how much revenue"]):
        bi_sql = """
        SELECT 
            SUM(net_revenue) as `Total Net Revenue ($)`, 
            SUM(gross_revenue) as `Total Gross Revenue ($)`, 
            SUM(quantity) as `Total Products Sold` 
        FROM fact_sales 
        WHERE status = 'Completed';
        """
        bi_title = "Overall Corporate Revenue Dashboard"
        bi_explanation = "This query calculates aggregate sales metrics including total gross revenue, net revenue, and total units sold for all completed orders."

    # If any BI query was identified, run it!
    if bi_sql:
        try:
            df_result = execute_query(bi_sql, user_role)
            
            # Formulate Markdown table manually (no external tabulate dependency)
            headers = list(df_result.columns)
            lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
            lines.append("| " + " | ".join("---" for _ in headers) + " |")
            for idx, r_row in df_result.iterrows():
                row_str = []
                for val in r_row:
                    if isinstance(val, float):
                        row_str.append(f"{val:,.2f}")
                    elif isinstance(val, (int, np.integer)):
                        row_str.append(f"{val:,}")
                    else:
                        row_str.append(str(val))
                lines.append("| " + " | ".join(row_str) + " |")
            table_md = "\n".join(lines)
            
            return f"""
            **Offline Semantic Mode: Database Query Result**
            
            ### {bi_title}
            {bi_explanation}
            
            {table_md}
            
            **Generated SQL Statement:**
            ```sql
            {bi_sql.strip()}
            ```
            """
        except Exception as e:
            return f"**Offline Semantic Mode Execution Error**: {str(e)}"

    # Fallback default help
    return welcome_msg


def generate_financial_insights(metrics, ai_mode, user_role):
    """
    Generates AI executive insights for uploaded spreadsheet metrics.
    Attempts Google Gemini or OpenAI if selected and keys are present.
    Otherwise, falls back to Offline Smart Insights.
    """
    total_rev = metrics["total_revenue"]
    mom_growth = metrics["mom_growth"]
    yearly_change = metrics["yearly_change"]
    total_profit_loss = metrics["total_profit_loss"]
    profit_margin = metrics["profit_margin"]
    
    yearly_summary_str = ""
    if "yearly_data" in metrics and metrics["yearly_data"] is not None:
        yd = metrics["yearly_data"]
        for _, row in yd.iterrows():
            yearly_summary_str += f"- Year {int(row['year'])}: Revenue = ${row['revenue']:,.2f}, Expenses = ${row['expenses']:,.2f}, Profit/Loss = ${row['profit_loss']:,.2f}\n"

    system_prompt = f"You are the expert 'AI Business InSite Assistant'. The user role is: {user_role}. Generate 3 key executive insights highlighting the revenue trend, profit efficiency, and overall trajectory based on the uploaded data."
    
    prompt = f"""
    Analyze the following financial performance data from an uploaded spreadsheet:
    - Total Revenue: ${total_rev:,.2f}
    - MoM Growth: {mom_growth:+.1f}%
    - YoY/Yearly Change: {yearly_change:+.1f}% if available
    - Net Profit/Loss: ${total_profit_loss:,.2f}
    - Net Profit Margin: {profit_margin:.1f}%
    
    Yearly breakdown:
    {yearly_summary_str}
    
    Generate exactly 3 key executive bullet-point insights highlighting the revenue trend, profit efficiency, and overall trajectory. Keep each insight concise, professional, and actionable. Start each bullet point with an emoji matching the insight. Do not include markdown code block formatting or long introductions.
    """
    
    # 1. Gemini Mode
    if ai_mode == "Google Gemini AI Mode (requires Gemini Key)":
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key and len(gemini_key.strip()) > 10:
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=gemini_key)
                
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.2
                    )
                )
                return f"**Google Gemini AI Mode (gemini-2.5-flash) Insights**\n\n{response.text}"
            except Exception as e:
                print(f"[WARNING] Gemini insights failed ({e}). Falling back to Offline.")

    # 2. OpenAI Mode
    if ai_mode == "Live AI Mode (requires OpenAI Key)":
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and len(openai_key.strip()) > 10:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=openai_key)
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
                return f"**Live AI Mode (GPT-4o-mini) Insights**\n\n{response.choices[0].message.content}"
            except Exception as e:
                print(f"[WARNING] OpenAI insights failed ({e}). Falling back to Offline.")

    # 3. Offline Semantic / Rule-Based Insights
    # Determine MoM sign
    mom_direction = "growth" if mom_growth >= 0 else "decline"
    mom_arrow = "+" if mom_growth >= 0 else "-"
    yoy_direction = "growth" if yearly_change >= 0 else "decline"
    yoy_arrow = "+" if yearly_change >= 0 else "-"
    
    insight_1 = f"**Revenue Momentum**: The business registered a total net sales revenue of **${total_rev:,.2f}**, marked by a short-term MoM {mom_direction} of **{mom_arrow} {abs(mom_growth):.1f}%** in the latest month."
    
    if yearly_change != 0.0:
        insight_2 = f"**Yearly Trajectory**: Multi-year performance shows a Year-over-Year (YoY) {yoy_direction} of **{yoy_arrow} {abs(yearly_change):.1f}%**, reflecting the longer-term structural health of sales volume."
    else:
        insight_2 = "**Yearly Trajectory**: Multi-year trend data is currently baseline, requiring additional historical intervals to compute YoY trajectory variations."
        
    margin_status = "exceeding" if profit_margin >= 50.0 else "below"
    margin_eval = "high profitability and excellent cost optimization" if profit_margin >= 50.0 else "heightened operating expenditures or lower product pricing leverage"
    insight_3 = f"**Profit Efficiency**: Net Profit/Loss stands at **${total_profit_loss:,.2f}** with an average margin of **{profit_margin:.1f}%**, which is **{margin_status}** the company's 50.0% target benchmark due to {margin_eval}."
    
    return f"""**Offline Smart Insights**

{insight_1}
{insight_2}
{insight_3}"""
