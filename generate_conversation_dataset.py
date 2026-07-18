# generate_conversation_dataset.py
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import mysql.connector
from sqlalchemy import create_engine

# Output directories
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_CSV = os.path.join(OUTPUT_DIR, "saas_sales_conversations.csv")
DATASET_JSON = os.path.join(OUTPUT_DIR, "saas_sales_embeddings.json")

# Database config (matches Mini-project4)
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Vinoth@0202",
    "database": "business_insite",
    "port": 3306
}

def fetch_real_transactions(num_records=500):
    """
    Fetches real transactions from raw_sales, raw_products, and raw_customers
    tables to align the conversation dataset with the actual database data.
    """
    print("[INFO] Attempting to fetch transactions from MySQL database...")
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        
        # Check if tables exist and have records
        cursor.execute("SHOW TABLES LIKE 'raw_sales';")
        if not cursor.fetchone():
            print("[INFO] raw_sales table not found. Using fallback synthetic generator.")
            cursor.close()
            conn.close()
            return None
            
        # Select a sample of 500 sales transactions joined with products and customers
        query = """
        SELECT 
            s.order_id, 
            s.customer_id, 
            s.product_id, 
            s.order_date, 
            s.quantity, 
            s.unit_price, 
            s.status,
            p.product_name, 
            p.category as product_category,
            c.customer_name, 
            c.region, 
            c.segment as customer_segment
        FROM raw_sales s
        LEFT JOIN raw_products p ON s.product_id = p.product_id
        LEFT JOIN raw_customers c ON s.customer_id = c.customer_id
        ORDER BY s.order_date DESC
        LIMIT %s;
        """
        cursor.execute(query, (num_records,))
        records = cursor.fetchall()
        
        # Also fetch support tickets to map objections
        cursor.execute("SELECT order_id, issue_type, root_cause FROM raw_issues;")
        tickets = {t["order_id"]: t for t in cursor.fetchall()}
        
        cursor.close()
        conn.close()
        
        if len(records) > 0:
            print(f"[SUCCESS] Fetched {len(records)} real sales transactions from database.")
            return {"records": records, "tickets": tickets}
        else:
            print("[INFO] raw_sales table is empty. Using fallback synthetic generator.")
            return None
            
    except Exception as e:
        print(f"[WARNING] Database fetch failed: {e}. Using fallback synthetic generator.")
        return None

def generate_saas_sales_conversations(num_records=500):
    np.random.seed(42)
    
    # 1. Try to fetch database records
    db_data = fetch_real_transactions(num_records)
    
    # Define vocabulary & templates
    reps = ["Sarah Jenkins", "Michael Chang", "Amanda Ross", "David Kim", "Jessica Taylor"]
    objections = ["Pricing", "Security/Compliance", "Custom Integration", "Competitor Comparison", "None"]
    industries = ["Fintech", "Healthcare", "E-commerce", "Edtech", "Logistics", "Cybersecurity"]
    
    # Region to industry mapping
    region_to_industry = {
        "North": "Logistics",
        "South": "Edtech",
        "East": "Fintech",
        "West": "Healthcare",
        "North America": "Cybersecurity",
        "Europe": "E-commerce",
        "Asia Pacific": "Fintech",
        "Latin America": "Logistics",
        "Imported": "E-commerce"
    }

    dialogue_starters = [
        "Hi, thanks for hopping on today. I'm excited to discuss how our solutions can fit your workflow.",
        "Hello! Thanks for taking the time. Let's look at your current database setups and operational bottlenecks.",
        "Hi there. I'd love to walk you through our platform integration and see how we can optimize your queries."
    ]

    objection_dialogues = {
        "Pricing": [
            "Prospect: The product fits our workflow, but the pricing tier is higher than our allocated budget.",
            "Rep: I understand. We can explore customizable licensing or set up a tiered rollout plan as you scale.",
            "Prospect: That structure would definitely help us get internal budget approval."
        ],
        "Security/Compliance": [
            "Prospect: We manage critical client data. Can you share details about your SOC2 compliance and encryption?",
            "Rep: Absolutely. We have full SOC2 Type II compliance and implement AES-256 data encryption at rest.",
            "Prospect: Perfect. Our compliance and security team will just need to review the formal audit report."
        ],
        "Custom Integration": [
            "Prospect: Our developers use a legacy infrastructure. Can your solution connect dynamically via custom APIs?",
            "Rep: Yes, we offer built-in REST adapters, custom database connectors, and webhook interfaces out of the box.",
            "Prospect: That's excellent. It will save our engineering team substantial development hours."
        ],
        "Competitor Comparison": [
            "Prospect: How does your tool compare to competitors in terms of caching speed and dashboard rendering?",
            "Rep: We focus on zero-code dashboard builds and instant server-side caching, cutting query costs by 40%.",
            "Prospect: Faster rendering and cost reduction are top priorities for our executive reporting."
        ],
        "None": [
            "Prospect: The setup looks very clean and the analytics interface is highly intuitive.",
            "Rep: Yes, our technical onboarding team can help connect your data sources and have you live in 48 hours.",
            "Prospect: That sounds great. We are ready to proceed with the contract setup."
        ]
    }

    data = []
    embeddings_list = []
    
    # If no DB data, fall back to purely synthetic records
    if not db_data:
        print(f"[INFO] Generating {num_records} synthetic B2B SaaS sales conversations...")
        start_date = datetime(2025, 7, 1)
        
        for i in range(num_records):
            conv_id = f"CONV_{i:04d}"
            rep = np.random.choice(reps)
            industry = np.random.choice(industries)
            company_size = np.random.choice(["Mid-Market", "Enterprise", "SMB"])
            objection = np.random.choice(objections)
            date = start_date + timedelta(days=int(np.random.randint(0, 365)))
            
            sales_effectiveness = float(np.clip(np.random.normal(0.7 if objection == "None" else 0.5, 0.15), 0.1, 1.0))
            customer_engagement = float(np.clip(np.random.normal(0.8 if company_size == "Enterprise" else 0.6, 0.15), 0.1, 1.0))
            
            success_score = (sales_effectiveness * 0.6) + (customer_engagement * 0.4)
            conversion_outcome = 1 if success_score > 0.55 else 0
            
            num_turns = np.random.randint(4, 9)
            prob_traj = [0.5]
            for t in range(num_turns - 2):
                prev = prob_traj[-1]
                drift = np.random.normal(0.08 if conversion_outcome == 1 else -0.08, 0.1)
                prob_traj.append(float(np.clip(prev + drift, 0.05, 0.95)))
            prob_traj.append(0.95 if conversion_outcome == 1 else 0.05)
            
            starter = np.random.choice(dialogue_starters)
            mid_parts = objection_dialogues[objection]
            closer = f"Rep: Great. Let's move forward with this package." if conversion_outcome == 1 else "Rep: I will check with my team and follow up."
            transcript = f"Rep: {starter}\n" + "\n".join(mid_parts) + f"\n{closer}"
            
            base_vector = np.zeros(3072)
            base_vector[objections.index(objection)] = 1.5
            base_vector[5 + industries.index(industry)] = 1.2
            base_vector[12 + conversion_outcome] = 2.0
            
            noise = np.random.normal(0, 0.1, 3072)
            raw_embedding = base_vector + noise
            norm = np.linalg.norm(raw_embedding)
            normalized_embedding = (raw_embedding / norm).tolist()
            
            pca_x = float(np.random.normal(1.5 if conversion_outcome == 1 else -1.5, 0.8))
            pca_y = float(np.random.normal(0.5 if objection == "Pricing" else -0.5, 0.6))
            pca_z = float(np.random.normal(1.0 if industry in ["Fintech", "Healthcare"] else -1.0, 0.7))
            
            record = {
                "conversation_id": conv_id,
                "sales_rep": rep,
                "customer_industry": industry,
                "company_size": company_size,
                "objection_type": objection,
                "date": date.strftime("%Y-%m-%d"),
                "sales_effectiveness": round(sales_effectiveness, 3),
                "customer_engagement": round(customer_engagement, 3),
                "conversion_outcome": conversion_outcome,
                "num_turns": num_turns,
                "probability_trajectory": json.dumps([round(p, 3) for p in prob_traj]),
                "pca_x": round(pca_x, 3),
                "pca_y": round(pca_y, 3),
                "pca_z": round(pca_z, 3),
                "transcript_snippet": transcript
            }
            data.append(record)
            embeddings_list.append({"conversation_id": conv_id, "embedding": normalized_embedding})
            
    else:
        # Generate based on real MySQL transactions!
        records = db_data["records"]
        tickets = db_data["tickets"]
        print(f"[INFO] Building dynamic conversation records from database data...")
        
        for idx, rec in enumerate(records):
            conv_id = f"CONV_{idx:04d}"
            rep = reps[idx % len(reps)]
            
            # Map region to industry
            region = str(rec["region"])
            industry = region_to_industry.get(region, np.random.choice(industries))
            
            # Map segment to company size
            segment = str(rec["customer_segment"])
            company_size = "Enterprise" if "enterprise" in segment.lower() else "Mid-Market" if "corporate" in segment.lower() else "SMB"
            
            # Determine conversion outcome from transaction status
            status = str(rec["status"])
            conversion_outcome = 1 if status.lower() == "completed" else 0
            
            # Determine objection type
            order_id = str(rec["order_id"])
            if order_id in tickets:
                ticket_type = tickets[order_id]["issue_type"]
                if ticket_type == "Billing Discrepancy":
                    objection = "Pricing"
                elif ticket_type == "Setup Failure":
                    objection = "Custom Integration"
                elif ticket_type == "Service Interruption":
                    objection = "Security/Compliance"
                else:
                    objection = "Custom Integration"
            else:
                if conversion_outcome == 1:
                    # Completed sales mostly had No objections (70%) or Pricing (15%) or Competitor (15%)
                    objection = np.random.choice(["None", "Pricing", "Competitor Comparison"], p=[0.70, 0.15, 0.15])
                else:
                    # Cancelled/refunded sales always had objections
                    objection = np.random.choice(["Pricing", "Security/Compliance", "Custom Integration", "Competitor Comparison"])
            
            # Date is the actual transaction date
            t_date = rec["order_date"]
            if isinstance(t_date, str):
                date_str = t_date
            else:
                date_str = t_date.strftime("%Y-%m-%d")
                
            # Sales effectiveness and engagement linked to outcome
            if conversion_outcome == 1:
                sales_effectiveness = float(np.random.uniform(0.65, 0.98))
                customer_engagement = float(np.random.uniform(0.60, 0.98))
            else:
                sales_effectiveness = float(np.random.uniform(0.10, 0.50))
                customer_engagement = float(np.random.uniform(0.10, 0.55))
                
            # Generate turn trajectory
            num_turns = np.random.randint(4, 9)
            prob_traj = [0.5]
            for t in range(num_turns - 2):
                prev = prob_traj[-1]
                drift = np.random.normal(0.08 if conversion_outcome == 1 else -0.08, 0.1)
                prob_traj.append(float(np.clip(prev + drift, 0.05, 0.95)))
            prob_traj.append(0.95 if conversion_outcome == 1 else 0.05)
            
            # Construct Dialogue Mentioning the Real Product Name
            p_name = str(rec["product_name"]) if rec["product_name"] else "our product"
            p_cat = str(rec["product_category"]) if rec["product_category"] else "SaaS Solution"
            
            starter = f"Rep: Hello, thank you for hopping on the call. I wanted to see how **{p_name}** can help support your **{company_size}** operations."
            
            if objection == "Pricing":
                objection_dialogue = (
                    f"Prospect: The capabilities of **{p_name}** fit our needs, but the licensing cost is higher than our budget.\n"
                    f"Rep: I understand. Since it's a premium **{p_cat}** service, we can look at a flexible billing cycle or phase the deployment.\n"
                    f"Prospect: That structure would definitely help us get internal budget approval."
                )
            elif objection == "Security/Compliance":
                objection_dialogue = (
                    f"Prospect: Since we are deploying **{p_name}** internally, our compliance team has strict data security requirements.\n"
                    f"Rep: Security is a priority. **{p_name}** offers full SOC2 Type II compliance and AES-256 data encryption at rest.\n"
                    f"Prospect: Great, we will just need to review the security audit documents."
                )
            elif objection == "Custom Integration":
                objection_dialogue = (
                    f"Prospect: We run custom legacy database servers. Can **{p_name}** integrate with our internal systems?\n"
                    f"Rep: Yes, **{p_name}** supports custom REST endpoints, pre-built SQL adapters, and secure webhook setups out of the box.\n"
                    f"Prospect: That's excellent. It will save our engineers considerable setup time."
                )
            elif objection == "Competitor Comparison":
                objection_dialogue = (
                    f"Prospect: How does **{p_name}** stand out compared to other **{p_cat}** services in the market?\n"
                    f"Rep: **{p_name}** offers 40% faster query caching and zero-code dashboard builds, saving both dev time and host bills.\n"
                    f"Prospect: The speed boost and hosting savings are strong selling points."
                )
            else:  # None
                objection_dialogue = (
                    f"Prospect: The setup walkthrough was very clear, and the interface seems extremely user-friendly.\n"
                    f"Rep: Excellent. We can set up **{p_name}** and connect your live tables within 48 hours.\n"
                    f"Prospect: Perfect. We are ready to proceed."
                )
                
            closer = f"Rep: Great. I'll send over the SLA and subscription contract for **{p_name}** today." if conversion_outcome == 1 else f"Rep: I will get our engineering team's feedback on this custom request for **{p_name}** and follow up."
            transcript = f"{starter}\n{objection_dialogue}\n{closer}"
            
            # Embeddings matching semantic shape
            base_vector = np.zeros(3072)
            base_vector[objections.index(objection)] = 1.5
            base_vector[5 + (industries.index(industry) if industry in industries else 0)] = 1.2
            base_vector[12 + conversion_outcome] = 2.0
            
            noise = np.random.normal(0, 0.1, 3072)
            raw_embedding = base_vector + noise
            norm = np.linalg.norm(raw_embedding)
            normalized_embedding = (raw_embedding / norm).tolist()
            
            pca_x = float(np.random.normal(1.5 if conversion_outcome == 1 else -1.5, 0.8))
            pca_y = float(np.random.normal(0.5 if objection == "Pricing" else -0.5, 0.6))
            pca_z = float(np.random.normal(1.0 if industry in ["Fintech", "Healthcare"] else -1.0, 0.7))
            
            record = {
                "conversation_id": conv_id,
                "sales_rep": rep,
                "customer_industry": industry,
                "company_size": company_size,
                "objection_type": objection,
                "date": date_str,
                "sales_effectiveness": round(sales_effectiveness, 3),
                "customer_engagement": round(customer_engagement, 3),
                "conversion_outcome": conversion_outcome,
                "num_turns": num_turns,
                "probability_trajectory": json.dumps([round(p, 3) for p in prob_traj]),
                "pca_x": round(pca_x, 3),
                "pca_y": round(pca_y, 3),
                "pca_z": round(pca_z, 3),
                "transcript_snippet": transcript
            }
            data.append(record)
            embeddings_list.append({"conversation_id": conv_id, "embedding": normalized_embedding})
            
    # Save files
    df = pd.DataFrame(data)
    df.to_csv(DATASET_CSV, index=False)
    print(f"[SUCCESS] Saved real-time tabular dataset metadata to: {DATASET_CSV}")
    
    with open(DATASET_JSON, 'w') as f:
        json.dump(embeddings_list, f)
    print(f"[SUCCESS] Saved real-time high-dimensional embeddings to: {DATASET_JSON}")
    
    return df, embeddings_list

def ingest_to_database(df, embeddings_list):
    try:
        print("[INFO] Ingesting real-time conversations into MySQL...")
        conn = mysql.connector.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            port=DB_CONFIG["port"]
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
        cursor.close()
        conn.close()
        
        # Connect directly to database
        db_conn = mysql.connector.connect(**DB_CONFIG)
        cursor = db_conn.cursor()
        
        # Drop tables in dependency order
        cursor.execute("DROP TABLE IF EXISTS dim_embeddings;")
        cursor.execute("DROP TABLE IF EXISTS fact_conversations;")
        db_conn.commit()
        
        # Create fact_conversations
        cursor.execute("""
        CREATE TABLE fact_conversations (
            conversation_id VARCHAR(50) PRIMARY KEY,
            sales_rep VARCHAR(100),
            customer_industry VARCHAR(100),
            company_size VARCHAR(50),
            objection_type VARCHAR(100),
            conversation_date DATE,
            sales_effectiveness DECIMAL(5,3),
            customer_engagement DECIMAL(5,3),
            conversion_outcome INT,
            num_turns INT,
            probability_trajectory TEXT,
            pca_x DECIMAL(10,3),
            pca_y DECIMAL(10,3),
            pca_z DECIMAL(10,3),
            transcript_snippet TEXT
        );
        """)
        
        # Create dim_embeddings
        cursor.execute("""
        CREATE TABLE dim_embeddings (
            conversation_id VARCHAR(50) PRIMARY KEY,
            embedding JSON,
            FOREIGN KEY (conversation_id) REFERENCES fact_conversations(conversation_id)
        );
        """)
        db_conn.commit()
        
        # Ingest fact data
        insert_fact_query = """
        INSERT INTO fact_conversations (
            conversation_id, sales_rep, customer_industry, company_size, objection_type,
            conversation_date, sales_effectiveness, customer_engagement, conversion_outcome,
            num_turns, probability_trajectory, pca_x, pca_y, pca_z, transcript_snippet
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        fact_records = []
        for r in df.to_dict(orient="records"):
            fact_records.append((
                r["conversation_id"], r["sales_rep"], r["customer_industry"], r["company_size"], r["objection_type"],
                r["date"], r["sales_effectiveness"], r["customer_engagement"], r["conversion_outcome"],
                r["num_turns"], r["probability_trajectory"], r["pca_x"], r["pca_y"], r["pca_z"], r["transcript_snippet"]
            ))
        cursor.executemany(insert_fact_query, fact_records)
        db_conn.commit()
        
        # Ingest embeddings data
        insert_emb_query = "INSERT INTO dim_embeddings (conversation_id, embedding) VALUES (%s, %s)"
        emb_records = []
        for e in embeddings_list:
            emb_records.append((e["conversation_id"], json.dumps(e["embedding"])))
        cursor.executemany(insert_emb_query, emb_records)
        db_conn.commit()
        
        print("[SUCCESS] MySQL database seeding complete.")
        cursor.close()
        db_conn.close()
    except Exception as e:
        print(f"[WARNING] Database ingestion skipped: {e}")

if __name__ == "__main__":
    df, embs = generate_saas_sales_conversations(500)
    ingest_to_database(df, embs)
