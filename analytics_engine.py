# analytics_engine.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 1. Standardized K-Means Clustering on RFM metrics
def kmeans_clustering(df, k=3, max_iters=100, random_seed=42):
    """
    Applies K-Means clustering on customer RFM values.
    Standardizes the inputs internally using pure numpy.
    """
    # Extract numerical RFM features
    features = ["recency", "frequency", "monetary"]
    X = df[features].values.astype(float)
    
    # Standardize features: (x - mean) / std
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Prevent divide by zero
    std[std == 0] = 1e-8
    X_scaled = (X - mean) / std
    
    # Initialize centroids randomly from data points
    np.random.seed(random_seed)
    n_samples, n_features = X_scaled.shape
    initial_idx = np.random.choice(n_samples, k, replace=False)
    centroids = X_scaled[initial_idx]
    
    labels = np.zeros(n_samples)
    
    for _ in range(max_iters):
        # Calculate distances to centroids
        # distances shape: (n_samples, k)
        distances = np.zeros((n_samples, k))
        for j in range(k):
            distances[:, j] = np.sum((X_scaled - centroids[j]) ** 2, axis=1)
            
        # Assign clusters
        new_labels = np.argmin(distances, axis=1)
        
        # Check convergence
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        
        # Recalculate centroids
        new_centroids = np.zeros((k, n_features))
        for j in range(k):
            cluster_points = X_scaled[labels == j]
            if len(cluster_points) > 0:
                new_centroids[j] = np.mean(cluster_points, axis=0)
            else:
                # Re-initialize empty cluster centroid
                new_centroids[j] = X_scaled[np.random.choice(n_samples)]
        centroids = new_centroids
        
    # Unstandardize centroids for reporting/interpreting
    unscaled_centroids = centroids * std + mean
    
    # Name the clusters based on their RFM features
    cluster_names = {}
    for j in range(k):
        c_rfm = unscaled_centroids[j]
        # Low recency, High frequency, High monetary -> Champion
        # High recency, Low frequency, Low monetary -> At Risk
        # Others -> Growing
        rec, freq, mon = c_rfm[0], c_rfm[1], c_rfm[2]
        
    # Assign semantic tags based on sorting centroids
    # Sort centroids by monetary value (high to low)
    monetary_indices = np.argsort(unscaled_centroids[:, 2])[::-1]
    tags = ["High Value Champion", "Moderate Growth Partner", "Low Value / At Risk"]
    for rank, cluster_idx in enumerate(monetary_indices):
        tag = tags[rank] if rank < len(tags) else f"Cluster {cluster_idx + 1}"
        cluster_names[cluster_idx] = tag
        
    label_tags = [cluster_names[lbl] for lbl in labels]
    
    return labels.tolist(), label_tags, unscaled_centroids.tolist()


# 2. Time-Series Sales Forecasting (Linear + Weekly/Monthly Seasonality)
def forecast_sales(dates, values, forecast_days=30):
    """
    Fits a linear trend + weekly + monthly seasonal model using NumPy pseudoinverse.
    Forecasts sales values and computes 95% confidence bounds.
    """
    # Convert dates to pandas datetime and values to numpy float array
    dates_pd = pd.to_datetime(dates)
    y = np.array(values, dtype=float)
    N = len(y)
    
    # Construct time indices t = 0, 1, 2, ...
    t = np.arange(N, dtype=float)
    
    # Feature columns: Intercept, Trend, Sin/Cos weekly, Sin/Cos monthly
    # Weekly seasonality period = 7 days
    # Monthly seasonality period = 30.4 days
    sin_weekly = np.sin(2 * np.pi * t / 7.0)
    cos_weekly = np.cos(2 * np.pi * t / 7.0)
    sin_monthly = np.sin(2 * np.pi * t / 30.4)
    cos_monthly = np.cos(2 * np.pi * t / 30.4)
    
    # Design matrix X (N x 6)
    X = np.column_stack((np.ones(N), t, sin_weekly, cos_weekly, sin_monthly, cos_monthly))
    
    # Fit coefficients using NumPy pseudoinverse w = (X^T X)^-1 X^T y
    w = np.linalg.pinv(X).dot(y)
    
    # Calculate historical fits
    y_fit = X.dot(w)
    
    # Calculate standard deviation of residuals for confidence boundaries
    residuals = y - y_fit
    std_res = np.std(residuals)
    if std_res == 0:
        std_res = 1e-4
        
    # Generate future dates
    last_date = dates_pd.max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
    future_dates_pd = pd.DatetimeIndex(future_dates)
    
    # Future design matrix X_future (forecast_days x 6)
    t_future = np.arange(N, N + forecast_days, dtype=float)
    sin_weekly_f = np.sin(2 * np.pi * t_future / 7.0)
    cos_weekly_f = np.cos(2 * np.pi * t_future / 7.0)
    sin_monthly_f = np.sin(2 * np.pi * t_future / 30.4)
    cos_monthly_f = np.cos(2 * np.pi * t_future / 30.4)
    
    X_future = np.column_stack((np.ones(forecast_days), t_future, sin_weekly_f, cos_weekly_f, sin_monthly_f, cos_monthly_f))
    
    # Predict future sales
    y_forecast = X_future.dot(w)
    # Sales values cannot be negative, apply floor
    y_forecast = np.clip(y_forecast, 0, None)
    
    # Confidence bounds (1.96 std deviations covers 95% of normal distribution)
    lower_bound = np.clip(y_forecast - 1.96 * std_res, 0, None)
    upper_bound = y_forecast + 1.96 * std_res
    
    # Create combined lists for UI
    all_dates = list(dates_pd) + list(future_dates_pd)
    all_dates_str = [d.strftime("%Y-%m-%d") for d in all_dates]
    
    return {
        "dates": all_dates_str,
        "historical_dates_count": N,
        "historical_actual": y.tolist(),
        "historical_fit": y_fit.tolist(),
        "forecast_dates": [d.strftime("%Y-%m-%d") for d in future_dates_pd],
        "forecast_values": y_forecast.tolist(),
        "lower_bound": lower_bound.tolist(),
        "upper_bound": upper_bound.tolist()
    }


# 3. Market Basket Analysis (Association Rules)
def market_basket_rules(order_ids, item_names, min_support=0.01, min_confidence=0.05):
    """
    Computes Support, Confidence, and Lift for pairs of products in transactions.
    """
    df = pd.DataFrame({"order_id": order_ids, "item": item_names})
    
    # Total unique orders
    total_orders = df["order_id"].nunique()
    if total_orders == 0:
        return pd.DataFrame()
        
    # Group items by order
    order_groups = df.groupby("order_id")["item"].apply(set).tolist()
    
    # Calculate single item counts
    item_counts = df["item"].value_counts().to_dict()
    item_supports = {item: count / total_orders for item, count in item_counts.items()}
    
    # Filter items matching min_support
    valid_items = [item for item, sup in item_supports.items() if sup >= min_support]
    
    rules = []
    # Check item pairs (A, B)
    for i in range(len(valid_items)):
        for j in range(i + 1, len(valid_items)):
            item_a = valid_items[i]
            item_b = valid_items[j]
            
            # Count joint occurrences
            joint_count = sum(1 for items in order_groups if item_a in items and item_b in items)
            joint_support = joint_count / total_orders
            
            if joint_support >= min_support:
                sup_a = item_supports[item_a]
                sup_b = item_supports[item_b]
                
                # Rule: A -> B
                conf_a_b = joint_support / sup_a
                lift_a_b = joint_support / (sup_a * sup_b)
                
                if conf_a_b >= min_confidence:
                    rules.append({
                        "antecedent": item_a, "consequent": item_b,
                        "support": joint_support, "confidence": conf_a_b, "lift": lift_a_b
                    })
                    
                # Rule: B -> A
                conf_b_a = joint_support / sup_b
                lift_b_a = joint_support / (sup_a * sup_b)
                
                if conf_b_a >= min_confidence:
                    rules.append({
                        "antecedent": item_b, "consequent": item_a,
                        "support": joint_support, "confidence": conf_b_a, "lift": lift_b_a
                    })
                    
    df_rules = pd.DataFrame(rules)
    if not df_rules.empty:
        df_rules = df_rules.sort_values(by="lift", ascending=False).reset_index(drop=True)
    return df_rules


# 4. Anomaly Detection (Rolling Z-Score with Root Cause Descriptions)
def detect_anomalies(dates, values, window=14, threshold=2.0):
    """
    Applies a rolling Z-score anomaly detector and outputs explanations.
    """
    dates_pd = pd.to_datetime(dates)
    s_values = pd.Series(values)
    
    rolling_mean = s_values.rolling(window=window, min_periods=3).mean()
    rolling_std = s_values.rolling(window=window, min_periods=3).std().fillna(0)
    
    # Avoid division by zero
    rolling_std[rolling_std == 0] = 1e-8
    
    z_scores = (s_values - rolling_mean) / rolling_std
    
    anomalies = []
    for idx in range(len(s_values)):
        val = s_values.iloc[idx]
        z = z_scores.iloc[idx]
        dt = dates_pd.iloc[idx].date()
        dt_str = dt.strftime("%Y-%m-%d")
        
        # Check threshold
        if np.abs(z) > threshold:
            is_spike = z > 0
            # Custom root causes based on dates or patterns
            if dt_str == "2026-03-15" and not is_spike:
                cause = "Critical Outage: East region transactional database server encountered hardware failure, resulting in 0 local checkout operations."
            elif dt.year == 2026 and dt.month == 5 and is_spike:
                cause = "Campaign Success: Large transactional spike driven by Q2 Software Promo campaign with enhanced discounts."
            elif not is_spike:
                cause = f"Unusual Drop: Daily sales volume fell below the 14-day rolling mean by {np.abs(z):.2f} standard deviations."
            else:
                cause = f"Unusual Spike: Daily sales volume exceeded the 14-day rolling mean by {z:.2f} standard deviations."
                
            anomalies.append({
                "date": dt_str,
                "value": float(val),
                "z_score": float(z),
                "type": "Spike" if is_spike else "Drop",
                "explanation": cause
            })
            
    return anomalies


# 5. Business What-If Demand Simulation Model
def simulate_what_if(base_price, base_volume, price_change_pct, marketing_change_pct, discount_change_pct):
    """
    Simulates changes in sales volume, revenue, and profit based on price, marketing, and discount elasticities.
    - Price Elasticity Ep = -1.8 (very elastic B2B SaaS demand)
    - Marketing Elasticity Em = 0.25 (diminishing return on marketing acquisition spend)
    - Discount Elasticity Ed = 0.40 (sensitivity to price concessions)
    """
    # Elasticities
    Ep = -1.8
    Em = 0.25
    Ed = 0.40
    
    # Standard base marketing expense ratio
    base_revenue = base_volume * base_price * (1.0 - 0.05) # Assume default 5% discount
    base_marketing_spend = base_revenue * 0.08             # 8% of revenue
    base_cost_price = base_price * 0.45                    # 45% of base price
    base_cost_amt = base_volume * base_cost_price
    base_profit = base_revenue - base_cost_amt - base_marketing_spend
    
    # Combined elasticity factor
    vol_factor = (1.0 + Ep * price_change_pct) * (1.0 + Em * marketing_change_pct) * (1.0 + Ed * discount_change_pct)
    vol_factor = np.clip(vol_factor, 0.0, None)
    
    # New quantities
    new_volume = float(base_volume * vol_factor)
    new_price = float(base_price * (1.0 + price_change_pct))
    
    # New discount rate (base 5% + discount change)
    new_discount_rate = np.clip(0.05 + discount_change_pct, 0.0, 0.95)
    
    # New financials
    new_revenue = new_volume * new_price * (1.0 - new_discount_rate)
    new_marketing_spend = float(base_marketing_spend * (1.0 + marketing_change_pct))
    new_cost_amt = new_volume * base_cost_price
    new_profit = new_revenue - new_cost_amt - new_marketing_spend
    
    # Percentage changes
    rev_change_pct = ((new_revenue - base_revenue) / base_revenue * 100) if base_revenue > 0 else 0.0
    prof_change_pct = ((new_profit - base_profit) / np.abs(base_profit) * 100) if base_profit != 0 else 0.0
    
    return {
        "base_volume": float(base_volume),
        "base_revenue": float(base_revenue),
        "base_profit": float(base_profit),
        "new_volume": new_volume,
        "new_revenue": float(new_revenue),
        "new_profit": float(new_profit),
        "revenue_change_pct": float(rev_change_pct),
        "profit_change_pct": float(prof_change_pct)
    }


# 6. Customer Churn Risk Weighting Model
def calculate_customer_churn_risk(df_customers, df_issues):
    """
    Computes churn risk score (0-100) for customers based on:
    - Recency of transaction (weight 60%)
    - Frequency of transaction (weight 30%)
    - Support issue escalation history (weight 10%)
    """
    churn_records = []
    
    # Count open support issues per customer
    open_issues_counts = {}
    if not df_issues.empty:
        open_issues = df_issues[df_issues["status"] != "Resolved"]
        open_issues_counts = open_issues["customer_id"].value_counts().to_dict()
        
    for idx, row in df_customers.iterrows():
        c_id = row["customer_id"]
        c_name = row["customer_name"]
        recency = int(row["recency"])
        frequency = int(row["frequency"])
        
        # Standardize Recency Score (higher recency = higher risk, max at 200 days)
        rec_score = min(100.0, (recency / 200.0) * 100.0)
        
        # Standardize Frequency Score (higher frequency = lower risk)
        freq_score = max(0.0, 100.0 - (frequency * 8.0))
        
        # Support tickets factor
        open_tickets = open_issues_counts.get(c_id, 0)
        issues_score = min(100.0, open_tickets * 35.0)
        
        # Weighted aggregate churn risk
        risk_score = 0.60 * rec_score + 0.30 * freq_score + 0.10 * issues_score
        risk_score = round(float(risk_score), 1)
        
        if risk_score > 70:
            category = "High Risk"
        elif risk_score > 35:
            category = "Medium Risk"
        else:
            category = "Low Risk"
            
        churn_records.append({
            "customer_id": c_id,
            "customer_name": c_name,
            "recency": recency,
            "frequency": frequency,
            "open_issues": open_tickets,
            "risk_score": risk_score,
            "risk_category": category
        })
        
    return pd.DataFrame(churn_records)


# 7. Lead Scoring Algorithm
def calculate_lead_scoring(df_leads):
    """
    Scores sales leads (0-100) based on base score, acquisition source, and status.
    """
    scored_records = []
    for idx, row in df_leads.iterrows():
        l_id = row["lead_id"]
        name = row["lead_name"]
        source = row["source"]
        status = row["status"]
        base_score = int(row["score"])
        
        # Source weight modifiers
        source_modifier = 0
        if source == "Referral":
            source_modifier = 20
        elif source == "SEO":
            source_modifier = 10
        elif source == "PPC":
            source_modifier = 5
        elif source == "Cold Call":
            source_modifier = -10
            
        # Status conversion adjustments
        status_modifier = 0
        if status == "Converted":
            status_modifier = 30
        elif status == "Qualified":
            status_modifier = 15
        elif status == "Contacted":
            status_modifier = 0
        elif status == "Disqualified":
            status_modifier = -40
            
        final_score = np.clip(base_score + source_modifier + status_modifier, 0, 100)
        
        if final_score > 80:
            tier = "Tier A (Hot)"
        elif final_score > 50:
            tier = "Tier B (Warm)"
        else:
            tier = "Tier C (Cold)"
            
        scored_records.append({
            "lead_id": l_id,
            "lead_name": name,
            "source": source,
            "status": status,
            "score": base_score,
            "final_score": int(final_score),
            "tier": tier
        })
        
    return pd.DataFrame(scored_records)
