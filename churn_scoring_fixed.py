"""
Churn Scoring Model — Revenue-Based
Loads ALL files from company_revenue_sheets/ (CSV + XLSX),
deduplicates by date, and scores churn risk based on revenue
trend, recency of decline, and volatility.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"

# ── Scoring config ─────────────────────────────────────────────────────────────
CONFIG = {
    # How many trailing days to use as "recent" window
    "recency_window": 30,
    # A month-over-month drop beyond this % triggers high recency risk
    "decline_threshold_pct": 15,
    # Weights must sum to 1.0
    "weight_recency":    0.40,   # how recently revenue fell
    "weight_trend":      0.35,   # overall direction (MoM slope)
    "weight_volatility": 0.25,   # revenue instability
    "thresholds": {
        "Low Risk":      (0,  30),
        "Medium Risk":   (31, 60),
        "High Risk":     (61, 80),
        "Critical Risk": (81, 100),
    },
}

RETENTION_PLAYBOOKS = {
    "Critical Risk": {
        "strategy": "Immediate executive review + emergency recovery plan + pricing intervention",
        "priority": 5,
    },
    "High Risk": {
        "strategy": "Urgent revenue audit + sales reactivation campaign + channel review",
        "priority": 4,
    },
    "Medium Risk": {
        "strategy": "Quarterly business review + promotional push + pipeline analysis",
        "priority": 3,
    },
    "Low Risk": {
        "strategy": "Monitor monthly + standard upsell cadence",
        "priority": 1,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_revenue_data() -> pd.DataFrame:
    """
    Reads every CSV and XLSX in company_revenue_sheets/.
    Deduplicates on Date so CSV/XLSX copies don't double-count.
    Adds a 'year' column derived from the date.
    """
    frames = []

    for path in sorted(DATA_DIR.iterdir()):
        if path.suffix == ".csv":
            df = pd.read_csv(path, parse_dates=["Date"])
        elif path.suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path, parse_dates=["Date"])
        else:
            continue
        df.columns = df.columns.str.strip()
        frames.append(df)
        print(f"  Loaded: {path.name}  ({len(df)} rows)")

    if not frames:
        raise FileNotFoundError(f"No CSV/XLSX files found in {DATA_DIR}")

    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["Date"])       # remove duplicate dates across files
        .sort_values("Date")
        .reset_index(drop=True)
    )
    combined["year"] = combined["Date"].dt.year.astype(str)
    print(f"\n  Combined: {len(combined)} unique daily records  "
          f"({combined['Date'].min().date()} to {combined['Date'].max().date()})\n")
    return combined


def build_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily revenue into monthly totals with MoM change."""
    monthly = (
        df.groupby(df["Date"].dt.to_period("M"))["Revenue"]
        .agg(total="sum", avg="mean", std="std", count="count")
        .reset_index()
    )
    monthly["Date"] = monthly["Date"].dt.to_timestamp()
    monthly["mom_change_pct"] = monthly["total"].pct_change() * 100
    monthly["rolling_3m_avg"] = monthly["total"].rolling(3).mean()
    return monthly


# ══════════════════════════════════════════════════════════════════════════════
# SCORING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def compute_recency_score(monthly: pd.DataFrame, cfg: dict) -> float:
    """
    Measures how recently revenue declined.
    Recent drops = high risk. Stable/growing recent months = low risk.
    """
    recent = monthly.dropna(subset=["mom_change_pct"]).tail(3)
    if recent.empty:
        return 0.5
    # Average of recent MoM changes (negative = declining)
    avg_recent_change = recent["mom_change_pct"].mean()
    threshold = cfg["decline_threshold_pct"]
    # Map: -threshold or worse → 1.0 risk, 0 or positive → 0.0 risk
    score = np.clip(-avg_recent_change / threshold, 0, 1)
    return float(score)


def compute_trend_score(monthly: pd.DataFrame) -> float:
    """
    Linear regression slope over all monthly totals.
    Negative slope = declining trend = higher risk.
    """
    y = monthly["total"].values
    x = np.arange(len(y))
    if len(x) < 2:
        return 0.5
    slope = np.polyfit(x, y, 1)[0]
    max_rev = y.max()
    # Normalise: slope as fraction of max revenue, then invert
    norm_slope = slope / max_rev
    score = np.clip(-norm_slope * 10, 0, 1)   # scale factor 10 keeps it 0-1
    return float(score)


def compute_volatility_score(monthly: pd.DataFrame) -> float:
    """
    Coefficient of variation across monthly totals.
    High volatility = unpredictable revenue = higher risk.
    """
    cv = monthly["total"].std() / monthly["total"].mean()
    # CV > 0.3 = very volatile, cap at 1.0
    score = np.clip(cv / 0.3, 0, 1)
    return float(score)


def compute_risk_score(monthly: pd.DataFrame, cfg: dict = CONFIG) -> dict:
    rec   = compute_recency_score(monthly, cfg)
    trend = compute_trend_score(monthly)
    vol   = compute_volatility_score(monthly)

    score = (
        rec   * cfg["weight_recency"] +
        trend * cfg["weight_trend"] +
        vol   * cfg["weight_volatility"]
    ) * 100

    return {
        "recency_component":     round(rec,   3),
        "trend_component":       round(trend, 3),
        "volatility_component":  round(vol,   3),
        "risk_score":            round(score, 1),
    }


def classify_risk(score: float) -> str:
    if score <= 30:
        return "Low Risk"
    elif score <= 60:
        return "Medium Risk"
    elif score <= 80:
        return "High Risk"
    else:
        return "Critical Risk"


# ══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_monthly_risk_table(monthly: pd.DataFrame, cfg: dict = CONFIG) -> pd.DataFrame:
    """Score each rolling 12-month window ending at each month."""
    records = []
    months = monthly["Date"].tolist()

    for i, end_date in enumerate(months):
        if i < 2:   # need at least 3 months for meaningful scoring
            continue
        window = monthly.iloc[: i + 1]
        scored = compute_risk_score(window, cfg)
        category = classify_risk(scored["risk_score"])
        playbook = RETENTION_PLAYBOOKS[category]

        records.append({
            "month":              end_date.strftime("%Y-%m"),
            "total_revenue":      round(window.iloc[-1]["total"], 2),
            "mom_change_pct":     round(window.iloc[-1]["mom_change_pct"], 1)
                                  if not pd.isna(window.iloc[-1]["mom_change_pct"]) else None,
            "recency_component":  scored["recency_component"],
            "trend_component":    scored["trend_component"],
            "volatility_component": scored["volatility_component"],
            "risk_score":         scored["risk_score"],
            "risk_category":      category,
            "priority":           playbook["priority"],
            "action":             playbook["strategy"],
        })

    return pd.DataFrame(records)


def print_report(daily: pd.DataFrame, monthly: pd.DataFrame,
                 risk_table: pd.DataFrame) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    sep = "=" * 80

    print(f"\n{sep}")
    print(f"  COMPANY REVENUE CHURN RISK REPORT  —  {now}")
    print(sep)

    # ── Overview ───────────────────────────────────────────────────────────────
    print("\n[OVERVIEW] DATASET SUMMARY\n")
    print(f"  Date range   : {daily['Date'].min().date()}  to  {daily['Date'].max().date()}")
    print(f"  Total days   : {len(daily)}")
    print(f"  Total revenue: ${daily['Revenue'].sum():,.0f}")
    print(f"  Daily avg    : ${daily['Revenue'].mean():,.0f}")
    print(f"  Daily high   : ${daily['Revenue'].max():,.0f}  ({daily.loc[daily['Revenue'].idxmax(), 'Date'].date()})")
    print(f"  Daily low    : ${daily['Revenue'].min():,.0f}  ({daily.loc[daily['Revenue'].idxmin(), 'Date'].date()})")

    # ── Monthly summary ────────────────────────────────────────────────────────
    print("\n[TASK A] MONTHLY REVENUE SUMMARY\n")
    disp = monthly[["Date", "total", "avg", "mom_change_pct"]].copy()
    disp.columns = ["Month", "Total ($)", "Daily Avg ($)", "MoM Change (%)"]
    disp["Month"] = disp["Month"].dt.strftime("%Y-%m")
    disp["Total ($)"] = disp["Total ($)"].map("${:,.0f}".format)
    disp["Daily Avg ($)"] = disp["Daily Avg ($)"].map("${:,.0f}".format)
    disp["MoM Change (%)"] = disp["MoM Change (%)"].map(
        lambda x: f"{x:+.1f}%" if pd.notna(x) else "—"
    )
    print(disp.to_string(index=False))

    # ── Risk scores ────────────────────────────────────────────────────────────
    print("\n[TASK B] MONTHLY RISK SCORE PROGRESSION\n")
    risk_disp = risk_table[["month", "total_revenue", "mom_change_pct",
                             "risk_score", "risk_category"]].copy()
    risk_disp.columns = ["Month", "Revenue ($)", "MoM %", "Risk Score", "Category"]
    risk_disp["Revenue ($)"] = risk_disp["Revenue ($)"].map("${:,.0f}".format)
    risk_disp["MoM %"] = risk_disp["MoM %"].map(
        lambda x: f"{x:+.1f}%" if pd.notna(x) else "—"
    )
    print(risk_disp.to_string(index=False))

    # ── Threshold distribution ─────────────────────────────────────────────────
    print("\n[TASK C] RISK TIER DISTRIBUTION ACROSS ALL MONTHS\n")
    for tier in ["Critical Risk", "High Risk", "Medium Risk", "Low Risk"]:
        members = risk_table[risk_table["risk_category"] == tier]["month"].tolist()
        bar = "#" * len(members)
        print(f"  {tier:<15} ({len(members):>2})  {bar}")
        if members:
            print(f"                         {', '.join(members)}")

    # ── Latest month action ────────────────────────────────────────────────────
    latest = risk_table.iloc[-1]
    print("\n[TASK D] CURRENT STATUS & RECOMMENDED ACTION\n")
    print(f"  Latest month : {latest['month']}")
    print(f"  Risk score   : {latest['risk_score']}")
    print(f"  Category     : {latest['risk_category']}")
    print(f"  Priority     : {'*' * int(latest['priority'])} ({int(latest['priority'])}/5)")
    print(f"  Action       : {latest['action']}")

    # ── Year-over-year ─────────────────────────────────────────────────────────
    print("\n[TASK E] YEAR-OVER-YEAR COMPARISON\n")
    yoy = daily.groupby("year")["Revenue"].agg(
        total="sum", avg="mean", peak="max", low="min"
    ).reset_index()
    # Add YoY growth column
    yoy["yoy_growth"] = yoy["total"].pct_change() * 100
    yoy["total"] = yoy["total"].map("${:,.0f}".format)
    yoy["avg"]   = yoy["avg"].map("${:,.0f}".format)
    yoy["peak"]  = yoy["peak"].map("${:,.0f}".format)
    yoy["low"]   = yoy["low"].map("${:,.0f}".format)
    yoy["yoy_growth"] = yoy["yoy_growth"].map(
        lambda x: f"{x:+.1f}%" if pd.notna(x) else "—"
    )
    print(yoy.to_string(index=False))

    print(f"\n{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
# DAILY RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_daily_scoring() -> pd.DataFrame:
    daily   = load_revenue_data()
    monthly = build_monthly_summary(daily)
    risk_table = generate_monthly_risk_table(monthly)

    print_report(daily, monthly, risk_table)

    out_path = f"churn_scores_{datetime.now().strftime('%Y%m%d')}.csv"
    risk_table.to_csv(out_path, index=False)
    print(f"  Scores saved -> {out_path}\n")

    return risk_table


if __name__ == "__main__":
    run_daily_scoring()
