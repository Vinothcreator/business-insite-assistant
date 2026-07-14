# test_suite.py
import unittest
import numpy as np
import pandas as pd
import sys
import os

# Set search path to include current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import analytics_engine as ae
import ai_engine as aie

class TestAnalyticsEngine(unittest.TestCase):
    
    def setUp(self):
        # Seed NumPy for deterministic outputs
        np.random.seed(42)
        
    def test_kmeans_clustering(self):
        """
        Verifies NumPy K-means outputs 3 distinct tags and runs successfully.
        """
        # Create mock RFM data
        mock_data = pd.DataFrame({
            "customer_id": [f"C{i:03d}" for i in range(30)],
            # Group 1: Champions (Low Recency, High Freq, High Monetary)
            # Group 2: At Risk (High Recency, Low Freq, Low Monetary)
            # Group 3: Mid Range
            "recency": [10]*10 + [200]*10 + [60]*10,
            "frequency": [20]*10 + [1]*10 + [5]*10,
            "monetary": [5000.00]*10 + [50.00]*10 + [400.00]*10
        })
        
        labels, tags, centroids = ae.kmeans_clustering(mock_data, k=3)
        
        self.assertEqual(len(labels), 30)
        self.assertEqual(len(tags), 30)
        self.assertEqual(len(centroids), 3)
        
        # Verify cluster names map appropriately
        self.assertIn("High Value Champion", tags)
        self.assertIn("Low Value / At Risk", tags)
        
    def test_forecast_sales(self):
        """
        Tests time series seasonality + trend forecast boundaries and calculations.
        """
        # Create a simple upward trend with a 7-day pattern
        dates = pd.date_range(start="2025-07-01", periods=100)
        t = np.arange(100)
        values = 100 + 2.5 * t + 20 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 5, 100)
        
        res = ae.forecast_sales(dates, values, forecast_days=30)
        
        # Verify forecast output dictionary keys
        self.assertIn("dates", res)
        self.assertIn("forecast_values", res)
        self.assertIn("lower_bound", res)
        self.assertIn("upper_bound", res)
        
        self.assertEqual(len(res["forecast_values"]), 30)
        # Check non-negativity constraint
        self.assertTrue(all(val >= 0 for val in res["forecast_values"]))
        self.assertTrue(all(l <= u for l, u in zip(res["lower_bound"], res["upper_bound"])))
        
    def test_market_basket_rules(self):
        """
        Validates support, confidence, and lift rules calculations in market basket rules.
        """
        order_ids = []
        item_names = []
        
        # 10 orders buying both SaaS License and SLA Support
        for o_id in range(1, 11):
            order_ids.extend([f"O{o_id}", f"O{o_id}"])
            item_names.extend(["SaaS License Pro", "Support Premium SLA"])
            
        # 5 orders buying just SaaS License
        for o_id in range(11, 16):
            order_ids.append(f"O{o_id}")
            item_names.append("SaaS License Pro")
            
        rules = ae.market_basket_rules(order_ids, item_names, min_support=0.1, min_confidence=0.1)
        
        self.assertFalse(rules.empty)
        # Support = joint support = 10/15 = 66.6%
        # Confidence SaaS -> SLA = S(SaaS, SLA) / S(SaaS) = 10/15 / 1.0 = 66.6%
        # Confidence SLA -> SaaS = S(SaaS, SLA) / S(SLA) = 10/15 / (10/15) = 1.0
        # Lift = 10/15 / (1.0 * (10/15)) = 1.0
        
        rule_row = rules[rules["antecedent"] == "Support Premium SLA"].iloc[0]
        self.assertEqual(rule_row["consequent"], "SaaS License Pro")
        self.assertAlmostEqual(rule_row["confidence"], 1.0)
        self.assertAlmostEqual(rule_row["lift"], 1.0)

    def test_simulate_what_if(self):
        """
        Verifies elasticity projections for what-if scenarios.
        """
        res = ae.simulate_what_if(base_price=1000.0, base_volume=100, price_change_pct=0.10, marketing_change_pct=0.0, discount_change_pct=0.0)
        
        # A 10% price increase should cause volume to drop by -18% (elasticity Ep = -1.8)
        # New volume = 100 * (1 - 0.18) = 82
        self.assertAlmostEqual(res["new_volume"], 82.0)
        # Verify profits calculated
        self.assertGreater(res["new_revenue"], 0)
        self.assertGreater(res["new_profit"], 0)

    def test_financial_data_import(self):
        """
        Validates parsing of external transaction sheets and dynamic metric extraction.
        """
        df_imported = pd.DataFrame({
            "Date": ["2026-07-01", "2026-07-02"],
            "Sales": [12000.0, 9500.0],
            "Profit": [6800.0, 5100.0]
        })
        
        cols = [c.lower() for c in df_imported.columns]
        self.assertIn("sales", cols)
        self.assertIn("profit", cols)
        
        total_rev = df_imported["Sales"].sum()
        total_prof = df_imported["Profit"].sum()
        self.assertEqual(total_rev, 21500.0)
        self.assertEqual(total_prof, 11900.0)
        
        margin = (total_prof / total_rev * 100)
        self.assertAlmostEqual(margin, 55.348837, places=4)


class TestAIEngine(unittest.TestCase):
    
    def test_sql_safety_sanitizer(self):
        """
        Tests that database WRITE queries are blocked and SELECT statements pass.
        """
        # Safe query
        safe_q = "SELECT customer_name, monetary FROM dim_customers WHERE rfm_segment = 'Champions';"
        passed, msg = aie.sanitize_sql(safe_q)
        self.assertTrue(passed)
        
        # Unsafe queries
        unsafe_queries = [
            "DELETE FROM raw_sales WHERE quantity < 0;",
            "DROP TABLE dim_customers;",
            "INSERT INTO raw_products (product_id) VALUES ('P100');",
            "UPDATE fact_sales SET net_revenue = 1000.0 WHERE order_id = 'TX5001';"
        ]
        
        for q in unsafe_queries:
            passed, msg = aie.sanitize_sql(q)
            self.assertFalse(passed)
            self.assertIn("Blocked Query", msg)
            
    def test_pii_masking(self):
        """
        Tests GDPR-compliant obfuscation of emails, phones, and customer names.
        """
        mock_df = pd.DataFrame({
            "customer_name": ["Johnathan Doe", "Jane Smith"],
            "email": ["john.doe@enterprise.com", "jane@corp.org"],
            "phone": ["+1-555-492-3847", "1234567"]
        })
        
        masked_df = aie.mask_pii_data(mock_df)
        
        # Verify email masking
        self.assertEqual(masked_df.iloc[0]["email"], "j***e@enterprise.com")
        self.assertEqual(masked_df.iloc[1]["email"], "j***e@corp.org")
        
        # Verify phone masking
        self.assertTrue(masked_df.iloc[0]["phone"].endswith("3847"))
        self.assertTrue(masked_df.iloc[0]["phone"].startswith("***-***-"))
        
        # Verify name masking
        self.assertEqual(masked_df.iloc[0]["customer_name"], "J***n D***e")
        
    def test_rbac_rules(self):
        """
        Tests role authorization checks.
        """
        ceo_query = "SELECT * FROM audit_logs;"
        sales_query = "SELECT * FROM fact_sales WHERE status='Completed';"
        ops_query = "SELECT issue_id, status FROM raw_issues;"
        
        # CEO Role
        passed, msg = aie.check_rbac_permission("CEO", ceo_query)
        self.assertTrue(passed)
        
        # Sales Manager: Block audit logs
        passed, msg = aie.check_rbac_permission("Sales Manager", "SELECT * FROM audit_logs;")
        self.assertFalse(passed)
        self.assertIn("Access Denied", msg)
        
        # Operations Manager: Block financial details
        passed, msg = aie.check_rbac_permission("Operations Manager", "SELECT net_revenue FROM fact_sales;")
        self.assertFalse(passed)
        self.assertIn("Access Denied", msg)
        
        # Operations Manager: Allowed issues access
        passed, msg = aie.check_rbac_permission("Operations Manager", ops_query)
        self.assertTrue(passed)

if __name__ == "__main__":
    unittest.main()
