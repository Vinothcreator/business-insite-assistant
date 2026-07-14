# AI Business InSite Assistant (Mini-project4)

This project is a premium, enterprise-ready Business Intelligence and AI assistant application designed to guide sales executives, managers, and operational teams through advanced data analytics, forecasting, customer value segmentation, and anomaly mitigations.

It implements a secure, performant, and compliant Star Schema Data Warehouse and runs completely out of the box using custom NumPy-based algorithms to ensure 100% compatibility across environments.

---

## 🏗️ Technology Stack & Connectivity
- **Backend / Language**: Python (3.14.3 virtual environment `.venv`)
- **Database**: Local MySQL server (port 3306)
- **Database Name**: `business_insite`
- **Credentials**: `root` / `Vinoth@0202`
- **Connectivity Libraries**: `mysql-connector-python` and `SQLAlchemy` (with URL-encoded password quote protection)
- **Frontend / UI**: Streamlit (configured with a professional dark-slate theme, custom typography via Google Fonts, custom CSS card accents, and responsive layout)
- **Visualizations**: Plotly Express & Plotly Graph Objects (dynamic line graphs, pie charts, 3D scatter plots, and correlation heatmaps)

---

## 📂 Project Structure
- [app.py](file:///c:/Users/Vinoth/OneDrive/Desktop/Data_Analysatics/Mini-project4/app.py): The main Streamlit user interface comprising Executive Dashboard, Advanced ML Tab, What-If Simulator, Anomaly Mitigation Board, and AI Chat Assistant.
- [db_setup.py](file:///c:/Users/Vinoth/OneDrive/Desktop/Data_Analysatics/Mini-project4/db_setup.py): Creates raw tables, seeds 12 months of daily transactional data with weekly cycles and seasonal anomalies, validates data quality (nulls, duplicates, bounds), compiles the Star Schema warehouse (`dim_customers` with computed RFM values, `fact_sales` with calculated profit margins), and populates metadata structures.
- [analytics_engine.py](file:///c:/Users/Vinoth/OneDrive/Desktop/Data_Analysatics/Mini-project4/analytics_engine.py): Implements mathematical calculations in pure NumPy & Pandas, including K-Means clustering, dual-seasonality Linear Regression forecasting, Market Basket rules, rolling Z-score anomalies, and what-if price elasticity simulations.
- [ai_engine.py](file:///c:/Users/Vinoth/OneDrive/Desktop/Data_Analysatics/Mini-project4/ai_engine.py): Coordinates SQL query sanitizers, PII obfuscation (names, emails, phones), role-based permissions access checks (CEO, Sales, Operations), semantic router, RAG search engine, and multi-agent collaborative chat loops.
- [test_suite.py](file:///c:/Users/Vinoth/OneDrive/Desktop/Data_Analysatics/Mini-project4/test_suite.py): Automated unit tests verifying ETL checks, predictive calculations, and security configurations.
- [.streamlit/config.toml](file:///c:/Users/Vinoth/OneDrive/Desktop/Data_Analysatics/Mini-project4/.streamlit/config.toml): Theme definitions setting up the dark-slate dashboard style.

---

## 🚀 Running the Project

### 1. Initialize the Database
Ensure your local MySQL service is running, then run the DB initialization script:
```bash
# From workspace root
.venv\Scripts\python Mini-project4/db_setup.py
```

### 2. Run Verification Tests
Verify all system logic, query safety blocks, PII masking rules, and predictive equations:
```bash
# From workspace root
.venv\Scripts\python -m unittest Mini-project4/test_suite.py
```

### 3. Launch the Dashboard
Start the Streamlit web application:
```bash
# From workspace root
.venv\Scripts\python -m streamlit run Mini-project4/app.py
```
Open `http://localhost:8501` in your browser.
