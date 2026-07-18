# import_kaggle_sales.py
import os
import sys
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from urllib.parse import quote_plus
from sqlalchemy import create_engine
import mysql.connector
import json
import shutil
import hashlib

# Paths
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(WORKSPACE_DIR, ".env")
DEST_DIR = os.path.join(WORKSPACE_DIR, "data", "raw")

def get_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def update_version_manifest(filename, status="Success", details=""):
    manifest_path = os.path.join(DEST_DIR, "version_manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}
            
    versions = manifest.get("versions", [])
    new_version = len(versions) + 1
    
    filepath = os.path.join(DEST_DIR, filename)
    if not os.path.exists(filepath):
        print(f"[WARNING] File {filepath} not found for version control.")
        return
        
    base, ext = os.path.splitext(filename)
    versioned_filename = f"{base}_v{new_version}{ext}"
    shutil.copy2(filepath, os.path.join(DEST_DIR, versioned_filename))
    
    file_size = os.path.getsize(filepath)
    file_hash = get_file_hash(filepath)
    
    versions.append({
        "version": new_version,
        "filename": filename,
        "versioned_filename": versioned_filename,
        "timestamp": datetime.now().isoformat(),
        "size_bytes": file_size,
        "sha256": file_hash,
        "status": status,
        "details": details
    })
    manifest["versions"] = versions
    manifest["latest_version"] = new_version
    manifest["last_updated"] = datetime.now().isoformat()
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
        
    print(f"[VERSION CONTROL] Saved {filename} as version {new_version} ({versioned_filename})")

def save_quality_report(dq_records):
    report_path = os.path.join(DEST_DIR, "quality_report.json")
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_checks": len(dq_records),
        "critical_issues": sum(1 for r in dq_records if r.get("severity") == "CRITICAL"),
        "warning_issues": sum(1 for r in dq_records if r.get("severity") == "WARNING"),
        "issues": dq_records
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"[QUALITY CHECKS] Quality report saved to {report_path}")

# Load environment variables
if os.path.exists(ENV_PATH):
    with open(ENV_PATH, "r") as f:
        for line in f:
            if not line.strip() or line.strip().startswith("#"):
                continue
            parts = line.strip().split("=", 1)
            if len(parts) == 2:
                os.environ[parts[0].strip()] = parts[1].strip()

# Database Config (matching db_setup.py)
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

def setup_kaggle_credentials():
    username = os.environ.get("KAGGLE_USERNAME", "")
    key = os.environ.get("KAGGLE_KEY", "")
    
    if not username or not key or "your_kaggle" in username or "your_kaggle" in key:
        print("[INFO] Kaggle API credentials are not set (or are placeholders) in the .env file.")
        print("We will skip the Kaggle CLI and download the dataset directly using a public fallback URL.")
        return False
        
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key
    print(f"[SUCCESS] Kaggle API Credentials detected (User: {username})")
    return True

def download_data(has_credentials=True):
    os.makedirs(DEST_DIR, exist_ok=True)
    has_downloaded_files = False
    
    if has_credentials:
        # Try rawanyasser42x/sales-analysis
        print("[INFO] Attempting to download outputs from Kaggle kernel: rawanyasser42x/sales-analysis...")
        cmd = [
            sys.executable, "-m", "kaggle", "kernels", "output", 
            "rawanyasser42x/sales-analysis", "-p", DEST_DIR
        ]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            downloaded = [f for f in os.listdir(DEST_DIR) if f.endswith(('.csv', '.xlsx')) and not f.startswith("Train_v") and "sample" not in f.lower()]
            if len(downloaded) > 0:
                print(f"[SUCCESS] Downloaded kernel outputs from rawanyasser42x: {downloaded}")
                has_downloaded_files = True
                for f in downloaded:
                    update_version_manifest(f, "Success", "Downloaded from rawanyasser42x/sales-analysis")
                    
        # Try aashirgurung/sales-analysis
        print("[INFO] Attempting to download outputs from Kaggle kernel: aashirgurung/sales-analysis...")
        cmd_aashir = [
            sys.executable, "-m", "kaggle", "kernels", "output", 
            "aashirgurung/sales-analysis", "-p", DEST_DIR
        ]
        print(f"Running: {' '.join(cmd_aashir)}")
        result_aashir = subprocess.run(cmd_aashir, capture_output=True, text=True)
        if result_aashir.returncode == 0:
            downloaded_aashir = [f for f in os.listdir(DEST_DIR) if f.endswith(('.csv', '.xlsx')) and not f.startswith("Train_v") and "sample" not in f.lower()]
            if len(downloaded_aashir) > 0:
                print(f"[SUCCESS] Downloaded kernel outputs from aashirgurung: {downloaded_aashir}")
                has_downloaded_files = True
                for f in downloaded_aashir:
                    update_version_manifest(f, "Success", "Downloaded from aashirgurung/sales-analysis")
                
        if not has_downloaded_files:
            print("[INFO] No outputs found or command failed. Falling back to downloading the source dataset 'brijbhushannanda1931/bigmart-sales-data' using Kaggle CLI...")
            ds_cmd = [
                sys.executable, "-m", "kaggle", "datasets", "download", 
                "-d", "brijbhushannanda1931/bigmart-sales-data", "-p", DEST_DIR, "--unzip"
            ]
            print(f"Running: {' '.join(ds_cmd)}")
            ds_result = subprocess.run(ds_cmd, capture_output=True, text=True)
            if ds_result.returncode == 0:
                downloaded = [f for f in os.listdir(DEST_DIR) if f.endswith(('.csv', '.xlsx')) and not f.startswith("Train_v") and "sample" not in f.lower()]
                print(f"[SUCCESS] Downloaded dataset files: {downloaded}")
                has_downloaded_files = True
                for f in downloaded:
                    update_version_manifest(f, "Success", "Downloaded from brijbhushannanda1931/bigmart-sales-data")
            else:
                print(f"[WARNING] Kaggle CLI dataset download failed: {ds_result.stderr}")
                
    if not has_downloaded_files:
        print("[INFO] Credentials missing or Kaggle CLI failed. Fetching BigMart Sales dataset directly from public GitHub repository...")
        import urllib.request
        fallback_url = "https://raw.githubusercontent.com/akki8087/Big-Mart-Sales/master/Train.csv"
        target_path = os.path.join(DEST_DIR, "Train.csv")
        try:
            print(f"Downloading {fallback_url} to {target_path}...")
            req = urllib.request.Request(fallback_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as response, open(target_path, "wb") as out_file:
                out_file.write(response.read())
            print("[SUCCESS] Successfully downloaded Train.csv directly from GitHub!")
            has_downloaded_files = True
            update_version_manifest("Train.csv", "Success", "Downloaded from GitHub fallback URL")
        except Exception as e:
            print(f"[ERROR] Failed to download from fallback URL: {e}")
            sys.exit(1)

def process_and_ingest():
    # 1. Locate downloaded sales analysis files
    csv_files = [os.path.join(DEST_DIR, f) for f in os.listdir(DEST_DIR) if f.lower().endswith('.csv') and not f.startswith("Train_v") and "sample" not in f.lower()]
    
    if len(csv_files) == 0:
        print("[ERROR] No sales CSV data files found!")
        sys.exit(1)
        
    # Read and consolidate all CSV files (handles multiple monthly CSVs)
    dfs = []
    for f in csv_files:
        try:
            print(f"[INFO] Reading sales data file: {f}")
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"[WARNING] Failed to read {f}: {e}")
            
    if len(dfs) == 0:
        print("[ERROR] No readable CSV files found!")
        sys.exit(1)
        
    df_raw = pd.concat(dfs, ignore_index=True)
    
    # Standardize column naming
    df_raw.columns = [c.strip() for c in df_raw.columns]
    df_raw = df_raw.dropna(how='all')
    
    # Check schema: detect if it's Electronics Sales or BigMart Sales
    cols_lower = [c.lower() for c in df_raw.columns]
    is_electronics = any(c in cols_lower for c in ['order id', 'order_id'])
    
    df_products = pd.DataFrame()
    df_customers = pd.DataFrame()
    df_sales = pd.DataFrame()
    
    start_date = datetime(2025, 7, 1).date()
    end_date = datetime(2026, 7, 31).date()
    regions = ["North", "South", "East", "West"]
    sales_channels = ["Online", "Enterprise Sales", "Partner Channel"]
    np.random.seed(42)
    
    if is_electronics:
        print("[INFO] Identified Electronics Store Sales Schema. Mapping columns...")
        
        # Map case-insensitively
        col_mapping = {}
        for col in df_raw.columns:
            cl = col.lower().replace(" ", "_")
            if cl == 'order_id':
                col_mapping[col] = 'order_id'
            elif cl in ['product', 'product_id']:
                col_mapping[col] = 'product_id'
            elif cl in ['quantity_ordered', 'quantity', 'qty']:
                col_mapping[col] = 'quantity'
            elif cl in ['price_each', 'unit_price', 'price']:
                col_mapping[col] = 'unit_price'
            elif cl in ['order_date', 'date']:
                col_mapping[col] = 'order_date'
            elif cl in ['purchase_address', 'address']:
                col_mapping[col] = 'purchase_address'
                
        df_raw = df_raw.rename(columns=col_mapping)
        
        # Remove header duplicates or rows missing critical keys
        df_raw = df_raw[df_raw['order_id'].notna() & (df_raw['order_id'].astype(str).str.lower() != 'order id')]
        df_raw['quantity'] = pd.to_numeric(df_raw['quantity'], errors='coerce')
        df_raw['unit_price'] = pd.to_numeric(df_raw['unit_price'], errors='coerce')
        df_raw = df_raw.dropna(subset=['quantity', 'unit_price', 'product_id', 'purchase_address'])
        
        # 2. Extract Products
        print("[INFO] Transforming Products Dimension (Electronics)...")
        df_prods = df_raw[['product_id', 'unit_price']].drop_duplicates('product_id')
        products_list = []
        for idx, row in df_prods.iterrows():
            p_id = row['product_id']
            price = float(row['unit_price'])
            
            p_lower = str(p_id).lower()
            if 'cable' in p_lower or 'charging' in p_lower:
                cat, sub_cat = 'Accessories', 'Cables'
            elif 'laptop' in p_lower or 'macbook' in p_lower:
                cat, sub_cat = 'Computers', 'Laptops'
            elif 'monitor' in p_lower or 'screen' in p_lower:
                cat, sub_cat = 'Monitors', 'Screens'
            elif 'phone' in p_lower or 'iphone' in p_lower:
                cat, sub_cat = 'Phones', 'Smartphones'
            elif 'headphones' in p_lower or 'earbuds' in p_lower:
                cat, sub_cat = 'Audio', 'Headphones'
            elif 'battery' in p_lower or 'batteries' in p_lower:
                cat, sub_cat = 'Accessories', 'Batteries'
            else:
                cat, sub_cat = 'Electronics', 'General'
                
            cost = price * 0.65 # Assume 35% margin
            products_list.append({
                "product_id": p_id,
                "product_name": p_id,
                "category": cat,
                "sub_category": sub_cat,
                "cost_price": round(cost, 2),
                "unit_price": round(price, 2)
            })
        df_products = pd.DataFrame(products_list)
        
        # Save a sample excel and csv in company_revenue_sheets for app uploads tab too
        df_company_rev_sample = df_raw[["order_id", "product_id", "unit_price", "quantity"]].copy()
        df_company_rev_sample.columns = ["TransactionID", "Product", "UnitPrice", "Quantity"]
        df_company_rev_sample["Revenue"] = df_company_rev_sample["UnitPrice"] * df_company_rev_sample["Quantity"]
        df_company_rev_sample = df_company_rev_sample.drop(columns=["Quantity"])
        
        revenue_vals = df_company_rev_sample["Revenue"].fillna(0.0)
        expenses_vals = revenue_vals * np.random.uniform(0.55, 0.75, size=len(revenue_vals))
        df_company_rev_sample["Expenses"] = expenses_vals.round(2)
        df_company_rev_sample["Date"] = [datetime(2025, 7, 1) + timedelta(days=i % 365) for i in range(len(df_company_rev_sample))]
        
        df_company_rev_sample.to_csv(os.path.join(DEST_DIR, "sample_company_revenue.csv"), index=False)
        df_company_rev_sample.to_excel(os.path.join(DEST_DIR, "sample_company_revenue.xlsx"), index=False)
        
        # 3. Extract Customers
        print("[INFO] Transforming Purchase Addresses into Customers Dimension...")
        df_custs = df_raw[['purchase_address']].drop_duplicates('purchase_address')
        customers_list = []
        mock_names = [
            "Apex Global", "Zenith Corp", "Nexus Tech", "Alpha Group", "Vanguard Industries",
            "Nova Solutions", "Quantum Holdings", "Matrix Systems", "Omni Ventures", "Beacon Labs",
            "Stellar Services", "Summit Media", "Delta Partners", "Horizon Logistics", "Pinnacle Brands"
        ]
        
        for idx, row in df_custs.reset_index(drop=True).iterrows():
            addr = row['purchase_address']
            cust_id = f"CUST_{idx + 1000:05d}"
            
            # Infer region from address
            state = "CA"
            parts = [p.strip() for p in addr.split(',')]
            if len(parts) >= 3:
                state_zip = parts[-1].split()
                if len(state_zip) >= 1:
                    state = state_zip[0]
            
            east_states = ["NY", "MA", "PA", "NJ", "CT", "RI", "ME", "NH", "VT"]
            west_states = ["CA", "WA", "OR", "NV", "AZ", "CO", "UT", "ID", "NM", "MT", "WY"]
            south_states = ["TX", "FL", "GA", "NC", "SC", "VA", "TN", "AL", "MS", "LA", "AR", "KY"]
            if state in east_states:
                region = "East"
            elif state in west_states:
                region = "West"
            elif state in south_states:
                region = "South"
            else:
                region = "North"
                
            cust_name = mock_names[idx % len(mock_names)] + f" Store {cust_id.split('_')[-1]}"
            email = f"manager_{cust_id.lower()}@retailops.com"
            phone = f"+1-555-888-{idx % 10000:04d}"
            segment = "Enterprise" if idx % 3 == 0 else "Corporate" if idx % 3 == 1 else "Mid-Market"
            acq_d = datetime(2025, 1, 1).date()
            status = "Active"
            
            customers_list.append({
                "customer_id": cust_id,
                "customer_name": cust_name,
                "email": email,
                "phone": phone,
                "region": region,
                "segment": segment,
                "acq_date": acq_d,
                "status": status,
                "purchase_address": addr
            })
        df_customers = pd.DataFrame(customers_list)
        addr_to_cust = {row['purchase_address']: (row['customer_id'], row['region']) for idx, row in df_customers.iterrows()}
        
        # 4. Generate Sales Transactions
        print("[INFO] Generating transactional dataset from Electronics Sales...")
        df_raw['order_date_parsed'] = pd.to_datetime(df_raw['order_date'], errors='coerce')
        default_dates = pd.date_range(start="2025-07-01", end="2026-07-31", periods=len(df_raw))
        df_raw['order_date_parsed'] = df_raw['order_date_parsed'].fillna(pd.Series(default_dates))
        
        generated_sales = []
        for idx, row in df_raw.iterrows():
            addr = row['purchase_address']
            p_id = row['product_id']
            qty = int(row['quantity'])
            mrp = float(row['unit_price'])
            order_id = row['order_id']
            
            cust_id, region = addr_to_cust.get(addr, (None, None))
            if not cust_id:
                cust_id = df_customers.iloc[idx % len(df_customers)]['customer_id']
                region = df_customers.iloc[idx % len(df_customers)]['region']
                
            channel = np.random.choice(sales_channels)
            order_date = row['order_date_parsed'].date()
            discount = np.random.choice([0.0, 0.05, 0.10, 0.15], p=[0.60, 0.20, 0.15, 0.05])
            
            status = "Completed"
            rand_val = np.random.random()
            if rand_val < 0.02:
                status = "Cancelled"
            elif rand_val < 0.05:
                status = "Refunded"
                
            generated_sales.append({
                "order_id": order_id,
                "customer_id": cust_id,
                "product_id": p_id,
                "order_date": order_date,
                "quantity": qty,
                "unit_price": mrp,
                "discount": discount,
                "sales_channel": channel,
                "region": region,
                "status": status
            })
        df_sales = pd.DataFrame(generated_sales)
        df_customers = df_customers.drop(columns=["purchase_address"])
        
    else:
        print("[INFO] Identified BigMart Sales Schema.")
        # 2. Extract Products
        print("[INFO] Transforming Products Dimension (BigMart)...")
        df_prods = df_raw[["Item_Identifier", "Item_Type", "Item_MRP", "Item_Fat_Content"]].drop_duplicates("Item_Identifier")
        products_list = []
        for idx, row in df_prods.iterrows():
            p_id = row["Item_Identifier"]
            cat = row["Item_Type"]
            price = float(row["Item_MRP"])
            fat = row["Item_Fat_Content"]
            
            sub_cat = "Regular" if "reg" in str(fat).lower() or "regular" in str(fat).lower() else "Low Fat"
            cost = price * 0.65
            name = f"{cat} ({p_id})"
            
            products_list.append({
                "product_id": p_id,
                "product_name": name,
                "category": cat,
                "sub_category": sub_cat,
                "cost_price": round(cost, 2),
                "unit_price": round(price, 2)
            })
        df_products = pd.DataFrame(products_list)
        
        # Save a sample excel and csv in company_revenue_sheets for app uploads tab too
        df_company_rev_sample = df_raw[["Item_Identifier", "Item_Type", "Item_MRP", "Item_Outlet_Sales"]].copy()
        df_company_rev_sample.columns = ["TransactionID", "Product", "UnitPrice", "Revenue"]
        
        revenue_vals = df_company_rev_sample["Revenue"].fillna(0.0)
        expenses_vals = revenue_vals * np.random.uniform(0.55, 0.75, size=len(revenue_vals))
        df_company_rev_sample["Expenses"] = expenses_vals.round(2)
        
        df_company_rev_sample["Date"] = [datetime(2025, 7, 1) + timedelta(days=i % 365) for i in range(len(df_company_rev_sample))]
        df_company_rev_sample.to_csv(os.path.join(DEST_DIR, "sample_company_revenue.csv"), index=False)
        df_company_rev_sample.to_excel(os.path.join(DEST_DIR, "sample_company_revenue.xlsx"), index=False)
        
        # 3. Extract Customers (Outlets)
        print("[INFO] Transforming Outlets into Customers Dimension...")
        df_outlets = df_raw[["Outlet_Identifier", "Outlet_Type", "Outlet_Location_Type", "Outlet_Establishment_Year"]].drop_duplicates("Outlet_Identifier")
        customers_list = []
        
        for idx, row in df_outlets.iterrows():
            o_id = row["Outlet_Identifier"]
            o_type = row["Outlet_Type"]
            loc_type = row["Outlet_Location_Type"]
            est_year = int(row["Outlet_Establishment_Year"])
            
            name = f"Outlet Store {o_id}"
            email = f"manager_{o_id.lower()}@bigmart.com"
            phone = f"+1-800-999-5{o_id.split('OUT')[-1]}"
            region = regions[idx % len(regions)]
            segment = o_type
            acq_d = datetime(est_year, 1, 1).date()
            status = "Active"
            
            customers_list.append({
                "customer_id": o_id,
                "customer_name": name,
                "email": email,
                "phone": phone,
                "region": region,
                "segment": segment,
                "acq_date": acq_d,
                "status": status
            })
        df_customers = pd.DataFrame(customers_list)
        
        # 4. Generate Sales Transactions
        print("[INFO] Generating transactional dataset from BigMart Sales...")
        generated_sales = []
        order_counter = 8000
        for idx, row in df_raw.iterrows():
            p_id = row["Item_Identifier"]
            cust_id = row["Outlet_Identifier"]
            tot_sales = float(row["Item_Outlet_Sales"])
            mrp = float(row["Item_MRP"])
            
            if pd.isna(tot_sales) or tot_sales <= 0:
                tot_sales = mrp * np.random.choice([1, 2, 3])
                
            qty = int(tot_sales / mrp)
            if qty <= 0:
                qty = 1
                
            tx_count = max(1, int(qty / 3))
            qty_left = qty
            
            cust_row = df_customers[df_customers["customer_id"] == cust_id].iloc[0]
            region = cust_row["region"]
            
            for tx_idx in range(tx_count):
                order_counter += 1
                order_id = f"TX{order_counter}"
                channel = np.random.choice(sales_channels)
                
                days_offset = np.random.randint(0, 395)
                order_date = start_date + timedelta(days=days_offset)
                
                if tx_idx == tx_count - 1:
                    order_qty = qty_left
                else:
                    order_qty = np.random.randint(1, min(4, qty_left + 1))
                
                qty_left -= order_qty
                discount = np.random.choice([0.0, 0.05, 0.10, 0.15], p=[0.60, 0.20, 0.15, 0.05])
                
                status = "Completed"
                rand_val = np.random.random()
                if rand_val < 0.02:
                    status = "Cancelled"
                elif rand_val < 0.05:
                    status = "Refunded"
                    
                generated_sales.append({
                    "order_id": order_id,
                    "customer_id": cust_id,
                    "product_id": p_id,
                    "order_date": order_date,
                    "quantity": order_qty,
                    "unit_price": mrp,
                    "discount": discount,
                    "sales_channel": channel,
                    "region": region,
                    "status": status
                })
                
                if qty_left <= 0:
                    break
        df_sales = pd.DataFrame(generated_sales)
        
    print(f"[SUCCESS] Generated {len(df_sales)} transactions representing real-time sales revenue history.")
    
    # 5. Leads and Issues
    print("[INFO] Generating operational Leads and Issues...")
    lead_sources = ["SEO", "PPC", "Referral", "Social", "Cold Call"]
    lead_statuses = ["Converted", "Qualified", "Contacted", "Disqualified"]
    generated_leads = []
    for i in range(200):
        l_id = f"L{1000 + i:04d}"
        name = f"Kaggle Lead Contact {i}"
        email = f"contact_{i}@kaggleleads.com"
        phone = f"+1-555-KAG-{i:03d}"
        source = np.random.choice(lead_sources)
        status = np.random.choice(lead_statuses, p=[0.3, 0.3, 0.2, 0.2])
        create_d = start_date + timedelta(days=np.random.randint(0, 360))
        score = np.random.randint(20, 100)
        generated_leads.append({
            "lead_id": l_id, "lead_name": name, "email": email, "phone": phone,
            "source": source, "status": status, "create_date": create_d, "score": score
        })
    df_leads = pd.DataFrame(generated_leads)
    
    generated_issues = []
    df_sales_valid = df_sales[df_sales["status"] == "Completed"].head(1000)
    issue_types = ["Late Delivery", "Service Interruption", "Billing Discrepancy", "Setup Failure"]
    issue_causes = {
        "Late Delivery": "Logistics Delay",
        "Service Interruption": "System Outage",
        "Billing Discrepancy": "Human Error",
        "Setup Failure": "Software Bug"
    }
    
    issue_counter = 100
    for idx, row in df_sales_valid.sample(n=100, random_state=123).iterrows():
        issue_counter += 1
        i_id = f"ISS{issue_counter}"
        o_id = row["order_id"]
        c_id = row["customer_id"]
        o_date = row["order_date"]
        
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
    
    # 6. Data Quality Auditor & Database Seeding
    print("[INFO] Performing Data Quality Checks...")
    dq_records = []
    
    # Check for invalid transactions in raw sales
    # 1. Null PKs or critical fields
    null_sales = df_sales[df_sales["order_id"].isna() | df_sales["customer_id"].isna() | df_sales["product_id"].isna()]
    for idx, row in null_sales.iterrows():
        dq_records.append({
            "table_name": "raw_sales", 
            "record_id": str(row["order_id"]), 
            "check_name": "Null Critical Fields",
            "log_message": "Critical field customer_id or product_id contains NULL. Record excluded.", 
            "severity": "CRITICAL"
        })
        
    # 2. Negative quantity or price
    neg_sales = df_sales[(df_sales["quantity"] < 0) | (df_sales["unit_price"] < 0)]
    for idx, row in neg_sales.iterrows():
        dq_records.append({
            "table_name": "raw_sales", 
            "record_id": str(row["order_id"]), 
            "check_name": "Negative Bounds Check",
            "log_message": f"Quantity ({row['quantity']}) or Price ({row['unit_price']}) is negative. Record excluded.", 
            "severity": "CRITICAL"
        })
        
    # 3. Duplicate Transaction Items
    dup_mask = df_sales.duplicated(subset=["order_id", "product_id"], keep="first")
    dup_sales = df_sales[dup_mask]
    for idx, row in dup_sales.iterrows():
        dq_records.append({
            "table_name": "raw_sales", 
            "record_id": f"{row['order_id']}_{row['product_id']}", 
            "check_name": "Duplicate Transaction Item",
            "log_message": f"Duplicate transaction item with order_id {row['order_id']} and product_id {row['product_id']} found. Record excluded.", 
            "severity": "CRITICAL"
        })
        
    save_quality_report(dq_records)
    
    # Exclude failed records to build clean tables
    valid_sales_mask = (
        (df_sales["quantity"] >= 0) & 
        (df_sales["unit_price"] >= 0) & 
        (df_sales["customer_id"].notna()) &
        (df_sales["product_id"].notna())
    )
    df_sales_clean = df_sales[valid_sales_mask]
    df_sales_clean = df_sales_clean.drop_duplicates(subset=["order_id", "product_id"])
    
    print("[INFO] Re-initializing Database connection...")
    conn = get_connection(use_db=False)
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
    cursor.close()
    conn.close()
    
    db_conn = get_connection(use_db=True)
    cursor = db_conn.cursor()
    
    print("[INFO] Dropping old tables...")
    tables_to_drop = [
        "agg_sales_monthly", "agg_sales_product_summary", "fact_sales",
        "dim_customers", "dim_products", "dim_dates", "dim_regions",
        "raw_sales", "raw_customers", "raw_products", "raw_leads", "raw_issues",
        "data_quality_logs"
    ]
    for t in tables_to_drop:
        cursor.execute(f"DROP TABLE IF EXISTS {t};")
    db_conn.commit()
    
    # Create Raw Tables
    print("[INFO] Re-creating operational raw tables...")
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
    cursor.execute("""
    CREATE TABLE data_quality_logs (
        log_id INT AUTO_INCREMENT PRIMARY KEY,
        table_name VARCHAR(100),
        record_id VARCHAR(100),
        check_name VARCHAR(100),
        log_message TEXT,
        severity VARCHAR(50),
        checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    db_conn.commit()
    
    password_encoded = quote_plus(DB_CONFIG['password'])
    engine_url = f"mysql+mysqlconnector://{DB_CONFIG['user']}:{password_encoded}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    engine = create_engine(engine_url)
    
    print("[INFO] Seeding raw tables with Kaggle data...")
    df_sales.to_sql("raw_sales", con=engine, if_exists="append", index=False)
    df_customers.to_sql("raw_customers", con=engine, if_exists="append", index=False)
    df_products.to_sql("raw_products", con=engine, if_exists="append", index=False)
    df_leads.to_sql("raw_leads", con=engine, if_exists="append", index=False)
    df_issues.to_sql("raw_issues", con=engine, if_exists="append", index=False)
    if dq_records:
        df_dq = pd.DataFrame(dq_records)
        df_dq.to_sql("data_quality_logs", con=engine, if_exists="append", index=False)
    db_conn.commit()
    
    # 7. ELT / Dimensional Modeling
    print("[INFO] Rebuilding dimensions...")
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
    curr = start_date
    while curr <= end_date:
        date_records.append({
            "date_id": curr, "day": curr.day, "month": curr.month, "year": curr.year,
            "quarter": (curr.month - 1) // 3 + 1, "day_name": curr.strftime("%A"),
            "is_weekend": curr.weekday() in [5, 6]
        })
        curr += timedelta(days=1)
    pd.DataFrame(date_records).to_sql("dim_dates", con=engine, if_exists="append", index=False)
    
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
    
    # Customers RFM Segmentation
    print("[INFO] Recomputing K-Means / RFM customer segmentation...")
    df_sales_completed = df_sales[df_sales["status"] == "Completed"]
    rfm_records = []
    for cust_id in df_customers["customer_id"]:
        cust_tx = df_sales_completed[df_sales_completed["customer_id"] == cust_id]
        if len(cust_tx) == 0:
            recency = 365
            frequency = 0
            monetary = 0.0
            segment = "Hibernating"
        else:
            last_date = cust_tx["order_date"].max()
            recency = (end_date - last_date).days
            frequency = len(cust_tx)
            monetary = float((cust_tx["quantity"] * cust_tx["unit_price"] * (1 - cust_tx["discount"])).sum())
            
            if recency <= 60 and frequency >= 20:
                segment = "Champions"
            elif recency <= 90 and frequency >= 10:
                segment = "Loyal Customers"
            elif recency <= 45 and frequency <= 4:
                segment = "Recent Customers"
            elif recency > 180 and frequency >= 10:
                segment = "At Risk"
            elif recency > 240:
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
    
    # Rebuild Fact Table
    print("[INFO] Rebuilding fact_sales table with profit metrics...")
    cursor.execute("SELECT region_id, region, channel FROM dim_regions;")
    regions_cache = {(r[1], r[2]): r[0] for r in cursor.fetchall()}
    
    cursor.execute("SELECT product_id, cost_price FROM dim_products;")
    products_cost_cache = {p[0]: float(p[1]) for p in cursor.fetchall()}
    
    fact_sales_records = []
    for idx, row in df_sales_clean.iterrows():
        o_id = row["order_id"]
        c_id = row["customer_id"]
        p_id = row["product_id"]
        o_date = row["order_date"]
        qty = row["quantity"]
        u_price = float(row["unit_price"])
        disc = float(row["discount"])
        status = row["status"]
        
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
        order_id VARCHAR(50),
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
        PRIMARY KEY (order_id, product_id),
        FOREIGN KEY (customer_id) REFERENCES dim_customers(customer_id),
        FOREIGN KEY (product_id) REFERENCES dim_products(product_id),
        FOREIGN KEY (date_id) REFERENCES dim_dates(date_id),
        FOREIGN KEY (region_id) REFERENCES dim_regions(region_id)
    );
    """)
    pd.DataFrame(fact_sales_records).to_sql("fact_sales", con=engine, if_exists="append", index=False)
    
    # Aggregated tables
    print("[INFO] Rebuilding aggregated tables (agg_sales_monthly, agg_sales_product_summary)...")
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
    cursor.close()
    db_conn.close()
    
    # Update conversation dataset if generate script exists
    gen_script = os.path.join(WORKSPACE_DIR, "generate_conversation_dataset.py")
    if os.path.exists(gen_script):
        print("[INFO] Re-generating conversational dataset from fresh Kaggle sales tables...")
        try:
            subprocess.run([sys.executable, gen_script], check=True)
            print("[SUCCESS] Conversational dataset regenerated.")
        except Exception as e:
            print(f"[WARNING] Failed to regenerate conversational dataset: {e}")
            
    print("[SUCCESS] Data Ingestion and ELT from Kaggle successfully complete!")

if __name__ == "__main__":
    has_creds = setup_kaggle_credentials()
    download_data(has_credentials=has_creds)
    process_and_ingest()
