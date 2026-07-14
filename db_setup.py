# db_setup.py
import mysql.connector
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import os

# Database configuration
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", "Vinoth@0202"),
    "database": os.environ.get("DB_NAME", "business_insite"),
    "port": int(os.environ.get("DB_PORT", 3306))
}

def get_connection(use_db=False):
    config = DB_CONFIG.copy()
    if not use_db:
        config.pop("database", None)
    return mysql.connector.connect(**config)

def setup_database():
    print("[INFO] Starting Database Setup for Mini-project4...")
    
    # 1. Create database if it doesn't exist
    conn = get_connection(use_db=False)
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
    print(f"[SUCCESS] Database '{DB_CONFIG['database']}' verified/created.")
    cursor.close()
    conn.close()
    
    # 2. Connect directly and clean up existing tables/views
    db_conn = get_connection(use_db=True)
    cursor = db_conn.cursor()
    
    print("[INFO] Cleaning up old tables and views...")
    # Drop in dependency order
    tables_to_drop = [
        "agg_sales_monthly", "agg_sales_product_summary", "fact_sales",
        "dim_customers", "dim_products", "dim_dates", "dim_regions",
        "raw_sales", "raw_customers", "raw_products", "raw_leads", "raw_issues",
        "data_quality_logs", "data_dictionary", "audit_logs"
    ]
    for table in tables_to_drop:
        cursor.execute(f"DROP TABLE IF EXISTS {table};")
    
    db_conn.commit()
    print("[SUCCESS] Cleanup completed.")
    
    # 3. Create tables
    print("[INFO] Creating raw operational tables...")
    
    # Raw Sales
    cursor.execute("""
    CREATE TABLE raw_sales (
        order_id VARCHAR(50),
        customer_id VARCHAR(50),
        product_id VARCHAR(50),
        order_date DATE,
        quantity INT,
        unit_price DECIMAL(10,2),
        discount DECIMAL(5,2),
        sales_channel VARCHAR(50),
        region VARCHAR(50),
        status VARCHAR(50)
    );
    """)
    
    # Raw Customers
    cursor.execute("""
    CREATE TABLE raw_customers (
        customer_id VARCHAR(50) PRIMARY KEY,
        customer_name VARCHAR(100),
        email VARCHAR(100),
        phone VARCHAR(50),
        region VARCHAR(50),
        segment VARCHAR(50),
        acq_date DATE,
        status VARCHAR(50)
    );
    """)
    
    # Raw Products
    cursor.execute("""
    CREATE TABLE raw_products (
        product_id VARCHAR(50) PRIMARY KEY,
        product_name VARCHAR(200),
        category VARCHAR(100),
        sub_category VARCHAR(100),
        cost_price DECIMAL(10,2),
        unit_price DECIMAL(10,2)
    );
    """)
    
    # Raw Leads
    cursor.execute("""
    CREATE TABLE raw_leads (
        lead_id VARCHAR(50) PRIMARY KEY,
        lead_name VARCHAR(100),
        email VARCHAR(100),
        phone VARCHAR(50),
        source VARCHAR(50),
        status VARCHAR(50),
        create_date DATE,
        score INT
    );
    """)
    
    # Raw Issues
    cursor.execute("""
    CREATE TABLE raw_issues (
        issue_id VARCHAR(50) PRIMARY KEY,
        order_id VARCHAR(50),
        customer_id VARCHAR(50),
        issue_type VARCHAR(100),
        priority VARCHAR(50),
        status VARCHAR(50),
        create_date DATE,
        resolved_date DATE,
        root_cause VARCHAR(200),
        action_taken VARCHAR(200)
    );
    """)
    
    # Data Quality Logs
    cursor.execute("""
    CREATE TABLE data_quality_logs (
        id INT AUTO_INCREMENT PRIMARY KEY,
        table_name VARCHAR(50),
        record_id VARCHAR(50),
        check_name VARCHAR(100),
        log_message TEXT,
        severity VARCHAR(20),
        log_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # Data Dictionary
    cursor.execute("""
    CREATE TABLE data_dictionary (
        id INT AUTO_INCREMENT PRIMARY KEY,
        table_name VARCHAR(50),
        column_name VARCHAR(50),
        data_type VARCHAR(50),
        description TEXT,
        lineage_source VARCHAR(100)
    );
    """)
    
    # Audit Logs
    cursor.execute("""
    CREATE TABLE audit_logs (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_role VARCHAR(50),
        user_query TEXT,
        generated_sql TEXT,
        execution_time_ms INT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status VARCHAR(50)
    );
    """)
    
    db_conn.commit()
    print("[SUCCESS] Raw and metadata tables created.")
    
    # 4. Generate Mock Data
    print("[INFO] Generating mock operational data (July 2025 to July 2026)...")
    start_date = datetime(2025, 7, 1).date()
    end_date = datetime(2026, 7, 31).date()
    num_days = (end_date - start_date).days + 1
    
    # Product catalog
    products_list = [
        {"id": "P001", "name": "CRM Enterprise Suite", "cat": "SaaS Software", "sub": "CRM", "cost": 500.00, "price": 1200.00},
        {"id": "P002", "name": "BI Analytics Platform", "cat": "SaaS Software", "sub": "Analytics", "cost": 400.00, "price": 950.00},
        {"id": "P003", "name": "AI Marketing Agent", "cat": "SaaS Software", "sub": "Marketing", "cost": 200.00, "price": 600.00},
        {"id": "P004", "name": "Security Firewall Shield", "cat": "SaaS Software", "sub": "Security", "cost": 300.00, "price": 750.00},
        {"id": "P005", "name": "Elastic VPS Compute Instance", "cat": "Cloud Infrastructure", "sub": "Compute", "cost": 100.00, "price": 250.00},
        {"id": "P006", "name": "Serverless GPU Cluster", "cat": "Cloud Infrastructure", "sub": "Compute", "cost": 600.00, "price": 1500.00},
        {"id": "P007", "name": "Managed Database Storage", "cat": "Cloud Infrastructure", "sub": "Storage", "cost": 80.00, "price": 180.00},
        {"id": "P008", "name": "Consulting Implementation Package", "cat": "Professional Services", "sub": "Implementation", "cost": 800.00, "price": 2000.00},
        {"id": "P009", "name": "Support Premium SLA", "cat": "Professional Services", "sub": "Support", "cost": 150.00, "price": 400.00}
    ]
    df_products = pd.DataFrame(products_list)
    df_products.columns = ["product_id", "product_name", "category", "sub_category", "cost_price", "unit_price"]
    
    # Customers
    regions = ["North", "South", "East", "West"]
    segments = ["Enterprise", "Corporate", "Mid-Market"]
    customer_names = [
        "Apex Global", "Zenith Corp", "Nexus Tech", "Alpha Group", "Vanguard Industries",
        "Nova Solutions", "Quantum Holdings", "Matrix Systems", "Omni Ventures", "Beacon Labs",
        "Stellar Services", "Summit Media", "Delta Partners", "Horizon Logistics", "Pinnacle Brands",
        "Legacy Finance", "Integrity Health", "Prestige Retail", "Eclipse Energy", "Synergy Consultants",
        "United Commerce", "First National", "Pacific Holdings", "Liberty Capital", "Dynamic Logistics",
        "Trident Security", "Catalyst Software", "True North", "Sentinel Group", "Blue Sky Holdings",
        "Infinite Horizons", "Pioneer Systems", "Fortress Security", "Aero Space", "Global Logistics",
        "National Steel", "Universal Trade", "Capital Ventures", "Core Engineering", "Advanced Metals"
    ]
    # Let's generate 100 customer records by combining words or expanding names
    generated_customers = []
    np.random.seed(42)
    for i in range(100):
        c_id = f"C{100 + i:03d}"
        if i < len(customer_names):
            base_name = customer_names[i]
        else:
            base_name = f"{np.random.choice(customer_names)} Sub_{i}"
        
        email = f"info@{base_name.lower().replace(' ', '')}.com"
        phone = f"+1-{np.random.randint(100, 999)}-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}"
        region = np.random.choice(regions)
        segment = np.random.choice(segments, p=[0.25, 0.45, 0.30])
        acq_d = start_date + timedelta(days=np.random.randint(0, 180))
        status = "Active" if np.random.random() < 0.90 else "Inactive"
        
        generated_customers.append({
            "customer_id": c_id, "customer_name": base_name, "email": email, "phone": phone,
            "region": region, "segment": segment, "acq_date": acq_d, "status": status
        })
    df_customers = pd.DataFrame(generated_customers)
    
    # Leads
    lead_sources = ["SEO", "PPC", "Referral", "Social", "Cold Call"]
    lead_statuses = ["Converted", "Qualified", "Contacted", "Disqualified"]
    generated_leads = []
    for i in range(150):
        l_id = f"L{1000 + i:04d}"
        name = f"Lead Contact {i}"
        email = f"contact_{i}@leadsample.com"
        phone = f"+1-{np.random.randint(100, 999)}-555-{np.random.randint(1000, 9999)}"
        source = np.random.choice(lead_sources)
        status = np.random.choice(lead_statuses, p=[0.3, 0.3, 0.2, 0.2])
        create_d = start_date + timedelta(days=np.random.randint(0, 360))
        score = np.random.randint(10, 99)
        generated_leads.append({
            "lead_id": l_id, "lead_name": name, "email": email, "phone": phone,
            "source": source, "status": status, "create_date": create_d, "score": score
        })
    df_leads = pd.DataFrame(generated_leads)
    
    # Sales Transactions
    sales_channels = ["Online", "Enterprise Sales", "Partner Channel"]
    generated_sales = []
    order_counter = 5000
    
    for day_idx in range(num_days):
        current_date = start_date + timedelta(days=day_idx)
        dow = current_date.weekday()
        
        # Growth trend multiplier (MoM 1.5% - 2%)
        trend_mult = 1.0 + (day_idx / 365) * 0.25
        
        # Weekly cycle: Higher sales Tue-Thu, lowest on Sat-Sun
        if dow in [5, 6]:
            base_orders = np.random.randint(2, 6)
        else:
            base_orders = np.random.randint(5, 12)
            
        base_orders = int(base_orders * trend_mult)
        
        # Region East outage anomaly on March 15, 2026
        is_east_outage = (current_date == datetime(2026, 3, 15).date())
        # May 2026 Promo Spike anomaly
        is_promo_spike = (current_date.year == 2026 and current_date.month == 5)
        
        if is_promo_spike:
            base_orders = int(base_orders * 2.2)
            
        for _ in range(base_orders):
            order_counter += 1
            order_id = f"TX{order_counter}"
            
            # Select customer
            cust = df_customers.sample(n=1).iloc[0]
            cust_id = cust["customer_id"]
            region = cust["region"]
            channel = np.random.choice(sales_channels)
            
            # Basic statuses
            status = "Completed"
            rand_status = np.random.random()
            if rand_status < 0.03:
                status = "Cancelled"
            elif rand_status < 0.08:
                status = "Refunded"
                
            # Outage override
            if is_east_outage and region == "East":
                continue # Skip transaction for East region on this day
                
            # Determine number of items in this order (to allow association rules!)
            num_items = int(np.random.choice([1, 2, 3], p=[0.70, 0.25, 0.05]))
            sampled_prods = df_products.sample(n=num_items)
            
            for _, prod in sampled_prods.iterrows():
                prod_id = prod["product_id"]
                qty = int(np.random.choice([1, 2, 3, 5], p=[0.70, 0.20, 0.08, 0.02]))
                unit_price = prod["unit_price"]
                
                # Promo spike features higher discounts
                if is_promo_spike:
                    discount = np.random.choice([0.15, 0.20, 0.30], p=[0.5, 0.3, 0.2])
                else:
                    discount = np.random.choice([0.0, 0.05, 0.10], p=[0.65, 0.25, 0.10])
                    
                generated_sales.append({
                    "order_id": order_id, "customer_id": cust_id, "product_id": prod_id,
                    "order_date": current_date, "quantity": qty, "unit_price": unit_price,
                    "discount": discount, "sales_channel": channel, "region": region, "status": status
                })
            
    # Inject bad rows for data quality checks
    corrupt_sales = [
        {"order_id": "TX_ERR1", "customer_id": "C001", "product_id": "P001", "order_date": start_date, "quantity": -5, "unit_price": 1200.00, "discount": 0.00, "sales_channel": "Online", "region": "North", "status": "Completed"},
        {"order_id": "TX_ERR2", "customer_id": "C002", "product_id": "P002", "order_date": start_date, "quantity": 1, "unit_price": -50.00, "discount": 0.00, "sales_channel": "Online", "region": "East", "status": "Completed"},
        {"order_id": "TX_ERR3", "customer_id": "C003", "product_id": None, "order_date": start_date, "quantity": 2, "unit_price": 950.00, "discount": 0.00, "sales_channel": "Online", "region": "South", "status": "Completed"},
        {"order_id": "TX_ERR4", "customer_id": None, "product_id": "P003", "order_date": start_date, "quantity": 1, "unit_price": 600.00, "discount": 0.00, "sales_channel": "Retail", "region": "West", "status": "Completed"},
    ]
    # Add duplicates
    corrupt_sales.append(generated_sales[0].copy())
    corrupt_sales[-1]["order_id"] = generated_sales[0]["order_id"] # Same ID
    
    df_sales = pd.DataFrame(generated_sales + corrupt_sales)
    
    # Issues
    generated_issues = []
    df_sales_valid = df_sales[~df_sales["order_id"].isin(["TX_ERR1", "TX_ERR2", "TX_ERR3", "TX_ERR4"])].head(1000)
    
    issue_types = ["Late Delivery", "Service Interruption", "Billing Discrepancy", "Setup Failure"]
    issue_causes = {
        "Late Delivery": "Logistics Delay",
        "Service Interruption": "System Outage",
        "Billing Discrepancy": "Human Error",
        "Setup Failure": "Software Bug"
    }
    
    issue_counter = 100
    for idx, row in df_sales_valid.sample(n=80, random_state=123).iterrows():
        issue_counter += 1
        i_id = f"ISS{issue_counter}"
        o_id = row["order_id"]
        c_id = row["customer_id"]
        o_date = row["order_date"]
        
        # Correlate West region late deliveries in June 2026
        if row["region"] == "West" and o_date.year == 2026 and o_date.month == 6:
            itype = "Late Delivery"
        else:
            itype = np.random.choice(issue_types)
            
        cause = issue_causes[itype]
        priority = np.random.choice(["Low", "Medium", "High", "Urgent"], p=[0.4, 0.35, 0.2, 0.05])
        status = np.random.choice(["Resolved", "In Progress", "Open"], p=[0.75, 0.15, 0.1])
        
        create_d = o_date + timedelta(days=np.random.randint(1, 4))
        resolved_d = create_d + timedelta(days=np.random.randint(1, 10)) if status == "Resolved" else None
        
        action = "None"
        if status == "Resolved":
            if itype == "Late Delivery":
                action = "Shipped replacement / credited shipping cost"
            elif itype == "Service Interruption":
                action = "Rebooted instance / issued service credit"
            elif itype == "Billing Discrepancy":
                action = "Corrected invoice / issued credit memo"
            elif itype == "Setup Failure":
                action = "Scheduled onboarding call / configured project settings"
                
        generated_issues.append({
            "issue_id": i_id, "order_id": o_id, "customer_id": c_id, "issue_type": itype,
            "priority": priority, "status": status, "create_date": create_d,
            "resolved_date": resolved_d, "root_cause": cause, "action_taken": action
        })
    df_issues = pd.DataFrame(generated_issues)
    
    # Save raw records to DB using SQLAlchemy
    password_encoded = quote_plus(DB_CONFIG['password'])
    engine_url = f"mysql+mysqlconnector://{DB_CONFIG['user']}:{password_encoded}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    engine = create_engine(engine_url)
    
    print("  -> Writing raw tables to MySQL database...")
    df_sales.to_sql("raw_sales", con=engine, if_exists="append", index=False)
    df_customers.to_sql("raw_customers", con=engine, if_exists="append", index=False)
    df_products.to_sql("raw_products", con=engine, if_exists="append", index=False)
    df_leads.to_sql("raw_leads", con=engine, if_exists="append", index=False)
    df_issues.to_sql("raw_issues", con=engine, if_exists="append", index=False)
    
    db_conn.commit()
    print("[SUCCESS] Operational raw tables populated.")
    
    # 5. ELT / Dimensional Modeling
    print("[INFO] Starting ELT Warehousing Layer (Star Schema & Data Cleaning)...")
    
    # Step A: Data Quality Auditor & Data Ingestion Pipeline
    dq_records = []
    
    # Check for invalid transactions in raw sales
    # 1. Null PKs or critical fields
    null_sales = df_sales[df_sales["order_id"].isna() | df_sales["customer_id"].isna() | df_sales["product_id"].isna()]
    for idx, row in null_sales.iterrows():
        dq_records.append({
            "table_name": "raw_sales", "record_id": str(row["order_id"]), "check_name": "Null Critical Fields",
            "log_message": f"Critical field customer_id or product_id contains NULL. Record excluded.", "severity": "CRITICAL"
        })
        
    # 2. Negative quantity or price
    neg_sales = df_sales[(df_sales["quantity"] < 0) | (df_sales["unit_price"] < 0)]
    for idx, row in neg_sales.iterrows():
        dq_records.append({
            "table_name": "raw_sales", "record_id": str(row["order_id"]), "check_name": "Negative Bounds Check",
            "log_message": f"Quantity ({row['quantity']}) or Price ({row['unit_price']}) is negative. Record excluded.", "severity": "CRITICAL"
        })
        
    # 3. Duplicate Order IDs
    dup_mask = df_sales.duplicated(subset=["order_id"], keep="first")
    dup_sales = df_sales[dup_mask]
    for idx, row in dup_sales.iterrows():
        dq_records.append({
            "table_name": "raw_sales", "record_id": str(row["order_id"]), "check_name": "Duplicate Primary Key",
            "log_message": f"Duplicate transaction with order_id {row['order_id']} found. Record excluded.", "severity": "CRITICAL"
        })
        
    if dq_records:
        df_dq = pd.DataFrame(dq_records)
        df_dq.to_sql("data_quality_logs", con=engine, if_exists="append", index=False)
        print(f"  -> Data Quality Auditor ran: Logged {len(dq_records)} issues.")
        
    # Exclude failed records to build clean tables
    valid_sales_mask = (
        (df_sales["quantity"] >= 0) & 
        (df_sales["unit_price"] >= 0) & 
        (~df_sales["order_id"].isin(dup_sales["order_id"])) &
        (df_sales["customer_id"].notna()) &
        (df_sales["product_id"].notna())
    )
    df_sales_clean = df_sales[valid_sales_mask]
    
    # Step B: Dimensions
    # dim_products
    cursor.execute("""
    CREATE TABLE dim_products (
        product_id VARCHAR(50) PRIMARY KEY,
        product_name VARCHAR(200),
        category VARCHAR(100),
        sub_category VARCHAR(100),
        cost_price DECIMAL(10,2),
        unit_price DECIMAL(10,2)
    );
    """)
    cursor.execute("INSERT INTO dim_products SELECT * FROM raw_products;")
    
    # dim_dates
    cursor.execute("""
    CREATE TABLE dim_dates (
        date_id DATE PRIMARY KEY,
        day INT,
        month INT,
        year INT,
        quarter INT,
        day_name VARCHAR(20),
        is_weekend BOOLEAN
    );
    """)
    date_records = []
    d_start = datetime(2025, 7, 1).date()
    d_end = datetime(2026, 7, 31).date()
    curr = d_start
    while curr <= d_end:
        date_records.append({
            "date_id": curr, "day": curr.day, "month": curr.month, "year": curr.year,
            "quarter": (curr.month - 1) // 3 + 1, "day_name": curr.strftime("%A"),
            "is_weekend": curr.weekday() in [5, 6]
        })
        curr += timedelta(days=1)
    pd.DataFrame(date_records).to_sql("dim_dates", con=engine, if_exists="append", index=False)
    
    # dim_regions
    cursor.execute("""
    CREATE TABLE dim_regions (
        region_id INT AUTO_INCREMENT PRIMARY KEY,
        region VARCHAR(50),
        channel VARCHAR(50)
    );
    """)
    region_combos = []
    for r in regions:
        for ch in sales_channels:
            region_combos.append({"region": r, "channel": ch})
    pd.DataFrame(region_combos).to_sql("dim_regions", con=engine, if_exists="append", index=False)
    
    # dim_customers (with calculated RFM and value segmentation)
    print("  -> Calculating RFM metrics for Customer Dimension...")
    df_sales_completed = df_sales_clean[df_sales_clean["status"] == "Completed"]
    
    rfm_records = []
    max_date = datetime(2026, 7, 31).date()
    for cust_id in df_customers["customer_id"]:
        cust_tx = df_sales_completed[df_sales_completed["customer_id"] == cust_id]
        if len(cust_tx) == 0:
            recency = 365 # Default/Inactive
            frequency = 0
            monetary = 0.0
            segment = "Hibernating"
        else:
            last_date = cust_tx["order_date"].max()
            recency = (max_date - last_date).days
            frequency = len(cust_tx)
            # monetary spend
            monetary = float((cust_tx["quantity"] * cust_tx["unit_price"] * (1 - cust_tx["discount"])).sum())
            
            # Simple rule-based RFM segmenter:
            if recency <= 45 and frequency >= 12:
                segment = "Champions"
            elif recency <= 60 and frequency >= 6:
                segment = "Loyal Customers"
            elif recency <= 30 and frequency <= 2:
                segment = "Recent Customers"
            elif recency > 150 and frequency >= 5:
                segment = "At Risk"
            elif recency > 180:
                segment = "Lost/Inactive"
            else:
                segment = "Needs Attention"
                
        rfm_records.append({
            "customer_id": cust_id,
            "recency": recency,
            "frequency": frequency,
            "monetary": monetary,
            "rfm_segment": segment
        })
    df_rfm = pd.DataFrame(rfm_records)
    df_customers_dw = df_customers.merge(df_rfm, on="customer_id")
    
    cursor.execute("""
    CREATE TABLE dim_customers (
        customer_id VARCHAR(50) PRIMARY KEY,
        customer_name VARCHAR(100),
        email VARCHAR(100),
        phone VARCHAR(50),
        region VARCHAR(50),
        segment VARCHAR(50),
        acq_date DATE,
        status VARCHAR(50),
        recency INT,
        frequency INT,
        monetary DECIMAL(12,2),
        rfm_segment VARCHAR(50)
    );
    """)
    df_customers_dw.to_sql("dim_customers", con=engine, if_exists="append", index=False)
    
    # Step C: Fact Sales Table
    print("  -> Populating Fact Table with computed profit metrics...")
    
    fact_sales_records = []
    # Fetch dim lookup caches
    cursor.execute("SELECT region_id, region, channel FROM dim_regions;")
    regions_cache = {(r[1], r[2]): r[0] for r in cursor.fetchall()}
    
    cursor.execute("SELECT product_id, cost_price FROM dim_products;")
    products_cost_cache = {p[0]: float(p[1]) for p in cursor.fetchall()}
    
    for idx, row in df_sales_clean.iterrows():
        o_id = row["order_id"]
        c_id = row["customer_id"]
        p_id = row["product_id"]
        o_date = row["order_date"]
        qty = row["quantity"]
        u_price = float(row["unit_price"])
        disc = float(row["discount"])
        status = row["status"]
        
        # Calculate
        gross_rev = qty * u_price
        disc_amt = gross_rev * disc
        net_rev = gross_rev - disc_amt if status == "Completed" else 0.00
        refund_amt = gross_rev - disc_amt if status == "Refunded" else 0.00
        
        p_cost = products_cost_cache.get(p_id, 0.0)
        cost_amt = qty * p_cost
        
        profit_amt = net_rev - cost_amt if status == "Completed" else -cost_amt
        profit_margin = (profit_amt / net_rev * 100) if (status == "Completed" and net_rev > 0) else 0.0
        
        region_id = regions_cache.get((row["region"], row["sales_channel"]), None)
        
        fact_sales_records.append({
            "order_id": o_id,
            "customer_id": c_id,
            "product_id": p_id,
            "date_id": o_date,
            "region_id": region_id,
            "quantity": qty,
            "unit_price": u_price,
            "discount": disc,
            "gross_revenue": gross_rev,
            "discount_amount": disc_amt,
            "net_revenue": net_rev,
            "refund_amount": refund_amt,
            "cost_amount": cost_amt,
            "profit_amount": profit_amt,
            "profit_margin": profit_margin,
            "status": status
        })
        
    cursor.execute("""
    CREATE TABLE fact_sales (
        order_id VARCHAR(50) PRIMARY KEY,
        customer_id VARCHAR(50),
        product_id VARCHAR(50),
        date_id DATE,
        region_id INT,
        quantity INT,
        unit_price DECIMAL(10,2),
        discount DECIMAL(5,2),
        gross_revenue DECIMAL(12,2),
        discount_amount DECIMAL(12,2),
        net_revenue DECIMAL(12,2),
        refund_amount DECIMAL(12,2),
        cost_amount DECIMAL(12,2),
        profit_amount DECIMAL(12,2),
        profit_margin DECIMAL(8,2),
        status VARCHAR(50),
        FOREIGN KEY (customer_id) REFERENCES dim_customers(customer_id),
        FOREIGN KEY (product_id) REFERENCES dim_products(product_id),
        FOREIGN KEY (date_id) REFERENCES dim_dates(date_id),
        FOREIGN KEY (region_id) REFERENCES dim_regions(region_id)
    );
    """)
    pd.DataFrame(fact_sales_records).to_sql("fact_sales", con=engine, if_exists="append", index=False)
    
    # Step D: Aggregated Tables for performance
    print("  -> Creating pre-calculated aggregate tables...")
    
    cursor.execute("""
    CREATE TABLE agg_sales_monthly (
        year INT,
        month INT,
        region VARCHAR(50),
        category VARCHAR(100),
        total_revenue DECIMAL(15,2),
        total_profit DECIMAL(15,2),
        order_count INT
    );
    """)
    cursor.execute("""
    INSERT INTO agg_sales_monthly
    SELECT 
        d.year, d.month, r.region, p.category,
        SUM(f.net_revenue), SUM(f.profit_amount), COUNT(f.order_id)
    FROM fact_sales f
    JOIN dim_dates d ON f.date_id = d.date_id
    JOIN dim_regions r ON f.region_id = r.region_id
    JOIN dim_products p ON f.product_id = p.product_id
    GROUP BY d.year, d.month, r.region, p.category;
    """)
    
    cursor.execute("""
    CREATE TABLE agg_sales_product_summary (
        product_id VARCHAR(50) PRIMARY KEY,
        product_name VARCHAR(200),
        category VARCHAR(100),
        total_qty_sold INT,
        total_net_sales DECIMAL(15,2),
        total_profit DECIMAL(15,2),
        average_discount DECIMAL(5,2)
    );
    """)
    cursor.execute("""
    INSERT INTO agg_sales_product_summary
    SELECT 
        p.product_id, p.product_name, p.category,
        SUM(f.quantity), SUM(f.net_revenue), SUM(f.profit_amount), AVG(f.discount)
    FROM fact_sales f
    JOIN dim_products p ON f.product_id = p.product_id
    GROUP BY p.product_id, p.product_name, p.category;
    """)
    
    db_conn.commit()
    print("[SUCCESS] Dimension and Fact tables populated.")
    
    # 6. Populate Data Dictionary / Governance definitions
    print("[INFO] Populating Data Dictionary...")
    dict_records = [
        {"table_name": "raw_sales", "column_name": "order_id", "data_type": "VARCHAR(50)", "description": "Primary key for raw transaction records", "lineage_source": "Source CRM/Checkout API"},
        {"table_name": "raw_sales", "column_name": "customer_id", "data_type": "VARCHAR(50)", "description": "FK linking transaction to customer", "lineage_source": "Source CRM/Checkout API"},
        {"table_name": "fact_sales", "column_name": "net_revenue", "data_type": "DECIMAL(12,2)", "description": "Actual net sales volume (Gross sales - discounts) for completed orders, 0 for cancelled/refunded", "lineage_source": "ELT calculated from quantity, unit_price, discount, and status"},
        {"table_name": "fact_sales", "column_name": "profit_amount", "data_type": "DECIMAL(12,2)", "description": "Net revenue minus product cost, tracks loss for cancellations and refunds", "lineage_source": "ELT calculated from revenue and dim_products cost price"},
        {"table_name": "dim_customers", "column_name": "rfm_segment", "data_type": "VARCHAR(50)", "description": "Customer behavioral value classification based on Recency, Frequency, Monetary metrics", "lineage_source": "ELT calculated customer purchase history"}
    ]
    pd.DataFrame(dict_records).to_sql("data_dictionary", con=engine, if_exists="append", index=False)
    
    db_conn.commit()
    cursor.close()
    db_conn.close()
    print("[SUCCESS] Data Ingestion and ELT setup completely successful!")

if __name__ == "__main__":
    setup_database()
