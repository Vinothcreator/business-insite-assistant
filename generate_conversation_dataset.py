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

def generate_saas_sales_conversations(num_records=500):
    print(f"[INFO] Generating {num_records} synthetic B2B SaaS sales conversations...")
    np.random.seed(42)
    
    # Pre-defined templates for conversations
    reps = ["Sarah Jenkins", "Michael Chang", "Amanda Ross", "David Kim", "Jessica Taylor"]
    industries = ["Fintech", "Healthcare", "E-commerce", "Edtech", "Logistics", "Cybersecurity"]
    company_sizes = ["Mid-Market", "Enterprise", "SMB"]
    objections = ["Pricing", "Security/Compliance", "Custom Integration", "Competitor Comparison", "None"]
    
    dialogue_starters = [
        "Hi, thanks for hopping on. I wanted to understand your current workflow with your CRM.",
        "Hello! Let's dive into your data pipelines and where you see the biggest bottlenecks.",
        "Thanks for taking the time today. I understand you're looking for a more secure analytics solution.",
        "Hi there. I'd love to walk you through our API integration capabilities and see how they fit your stack."
    ]
    
    objection_dialogues = {
        "Pricing": [
            "Prospect: The platform looks great, but the enterprise pricing tier is higher than our budget.",
            "Rep: I understand. We can look at customized billing, or start with a standard tier and expand as you scale.",
            "Prospect: That sounds reasonable. What does the custom package include?"
        ],
        "Security/Compliance": [
            "Prospect: We handle highly regulated health data. Do you have SOC2 Type II and HIPAA compliance?",
            "Rep: Yes, absolutely. We can provide our SOC2 report and sign a BAA for HIPAA compliance.",
            "Prospect: Excellent, our compliance officer will need to review that document."
        ],
        "Custom Integration": [
            "Prospect: We use a legacy SQL server and custom internal APIs. Can your tool connect directly?",
            "Rep: We support custom DB connectors and REST API endpoints out of the box, plus Webhooks.",
            "Prospect: Good to know. That saves us a lot of developer hours."
        ],
        "Competitor Comparison": [
            "Prospect: How does your analytics performance compare to competitors like Snowflake or Databricks?",
            "Rep: We focus on zero-code BI dashboards and instant caching, saving up to 40% on querying costs.",
            "Prospect: Caching is key for us. That would speed up our internal executive reporting."
        ],
        "None": [
            "Prospect: The interface is very intuitive. The setup seems straightforward.",
            "Rep: Yes, our onboarding team can have your database connected and dashboards running in 48 hours.",
            "Prospect: Perfect, we are ready to proceed with the trial."
        ]
    }
    
    closing_dialogues = [
        "Rep: Great. I'll send over the contract and security documents. Let's schedule a kickoff next Tuesday.",
        "Rep: Let's set up a 14-day trial environment with a sample dataset to prove the speed metrics.",
        "Rep: I will compile a custom quote for the enterprise package and send it by end of day.",
        "Rep: Thanks for the feedback. I'll check with our engineering team on the custom API support and follow up."
    ]

    data = []
    embeddings_list = []
    
    start_date = datetime(2025, 7, 1)
    
    for i in range(num_records):
        conv_id = f"CONV_{i:04d}"
        rep = np.random.choice(reps)
        industry = np.random.choice(industries)
        company_size = np.random.choice(company_sizes)
        objection = np.random.choice(objections)
        date = start_date + timedelta(days=int(np.random.randint(0, 365)))
        
        # Performance metrics
        sales_effectiveness = float(np.clip(np.random.normal(0.7 if objection == "None" else 0.5, 0.15), 0.1, 1.0))
        customer_engagement = float(np.clip(np.random.normal(0.8 if company_size == "Enterprise" else 0.6, 0.15), 0.1, 1.0))
        
        # Calculate conversion outcome based on engagement and effectiveness
        success_score = (sales_effectiveness * 0.6) + (customer_engagement * 0.4)
        conversion_outcome = 1 if success_score > 0.55 else 0
        
        # Create probability trajectory (simulating turn-by-turn success probability)
        num_turns = np.random.randint(4, 9)
        prob_traj = [0.5]  # Start at 50%
        for t in range(num_turns - 2):
            prev = prob_traj[-1]
            # Drift direction depends on conversion outcome
            drift = np.random.normal(0.08 if conversion_outcome == 1 else -0.08, 0.1)
            prob_traj.append(float(np.clip(prev + drift, 0.05, 0.95)))
        prob_traj.append(0.95 if conversion_outcome == 1 else 0.05)  # Final turn probability
        
        # Build Transcript
        starter = np.random.choice(dialogue_starters)
        mid_parts = objection_dialogues[objection]
        closer = np.random.choice(closing_dialogues)
        transcript = f"Rep: {starter}\n" + "\n".join(mid_parts) + f"\n{closer}"
        
        # Generate synthetic 3072D Embedding using normal distribution
        # Base vector centered around features representing the objection, industry, and outcome
        base_vector = np.zeros(3072)
        # Seed embeddings with semantic dimensions
        base_vector[objections.index(objection)] = 1.5  # Objections dimensional vector
        base_vector[5 + industries.index(industry)] = 1.2  # Industry dimensional vector
        base_vector[12 + conversion_outcome] = 2.0  # Outcome dimensional vector
        
        # Add random noise
        noise = np.random.normal(0, 0.1, 3072)
        raw_embedding = base_vector + noise
        # Normalize embedding to unit length
        norm = np.linalg.norm(raw_embedding)
        normalized_embedding = (raw_embedding / norm).tolist()
        
        # Generate 3D PCA projection coordinates for dashboard plotting
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
        
        embeddings_list.append({
            "conversation_id": conv_id,
            "embedding": normalized_embedding
        })

    df = pd.DataFrame(data)
    df.to_csv(DATASET_CSV, index=False)
    print(f"[SUCCESS] Saved tabular dataset metadata to: {DATASET_CSV}")
    
    with open(DATASET_JSON, 'w') as f:
        json.dump(embeddings_list, f)
    print(f"[SUCCESS] Saved high-dimensional 3072D embeddings to: {DATASET_JSON}")
    
    return df, embeddings_list

def ingest_to_database(df, embeddings_list):
    try:
        print("[INFO] Setting up database schema and ingesting data into MySQL...")
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
        print(" -> Creating table fact_conversations...")
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
        
        # Create dim_embeddings for high-dimensional search (optimized JSON column)
        print(" -> Creating table dim_embeddings...")
        cursor.execute("""
        CREATE TABLE dim_embeddings (
            conversation_id VARCHAR(50) PRIMARY KEY,
            embedding JSON,
            FOREIGN KEY (conversation_id) REFERENCES fact_conversations(conversation_id)
        );
        """)
        db_conn.commit()
        
        # Ingest fact data
        print(" -> Ingesting conversation fact data...")
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
        print(" -> Ingesting 3072D embeddings vector database tables...")
        insert_emb_query = "INSERT INTO dim_embeddings (conversation_id, embedding) VALUES (%s, %s)"
        emb_records = []
        for e in embeddings_list:
            emb_records.append((e["conversation_id"], json.dumps(e["embedding"])))
        cursor.executemany(insert_emb_query, emb_records)
        db_conn.commit()
        
        print("[SUCCESS] Ingestion to MySQL completely successful!")
        cursor.close()
        db_conn.close()
    except Exception as e:
        print(f"[ERROR] Database ingestion failed: {e}")

if __name__ == "__main__":
    df, embs = generate_saas_sales_conversations(500)
    ingest_to_database(df, embs)
