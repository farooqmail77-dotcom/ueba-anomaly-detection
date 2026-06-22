"""
alerts.py
Alert formatting and output helpers.
Converts raw scored DataFrames into human-readable alert records,
supports output to CSV, JSON, and a rich console table.
"""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
import pandas as pd


# ── Alert severity thresholds ─────────────────────────────────────────────────
SEVERITY_MAP = {
    (0.85, 1.01): "CRITICAL",
    (0.70, 0.85): "HIGH",
    (0.55, 0.70): "MEDIUM",
    (0.00, 0.55): "INFO",
}


def _severity(risk_score: float) -> str:
    for (lo, hi), label in SEVERITY_MAP.items():
        if lo <= risk_score < hi:
            return label
    return "INFO"


def build_alert_records(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to alerted events and enrich with severity and human-readable summary.

    Parameters
    ----------
    scored_df : pd.DataFrame
        Output of UEBADetector.score().

    Returns
    -------
    pd.DataFrame
        One row per alert with columns:
        alert_id, timestamp, user, risk_score, severity,
        top_features, ground_truth_type, summary
    """
    alerts = scored_df[scored_df["alert"] == 1].copy().reset_index(drop=True)
    if alerts.empty:
        return alerts

    alerts["alert_id"] = [f"UEBA-{i+1:05d}" for i in range(len(alerts))]
    alerts["severity"] = alerts["risk_score"].apply(_severity)
    alerts["summary"] = alerts.apply(_build_summary, axis=1)

    cols = [
        "alert_id", "timestamp", "user", "risk_score", "severity",
        "if_score", "xgb_score", "top_features",
        "anomaly_type",  # ground truth — not available in production
        "summary",
    ]
    return alerts[[c for c in cols if c in alerts.columns]]


def _build_summary(row) -> str:
    ts = row["timestamp"]
    if isinstance(ts, (pd.Timestamp, datetime)):
        ts_str = ts.strftime("%Y-%m-%d %H:%M")
    else:
        ts_str = str(ts)
    return (
        f"User '{row['user']}' triggered {row['severity']} alert "
        f"(risk={row['risk_score']:.3f}) at {ts_str}. "
        f"Top drivers: {row.get('top_features', 'N/A')}."
    )


# ── Output writers ─────────────────────────────────────────────────────────────

def save_alerts_csv(alert_df: pd.DataFrame, path: str = "reports/alerts.csv"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    alert_df.to_csv(path, index=False)
    print(f"Alerts saved → {path}  ({len(alert_df)} rows)")


def save_alerts_json(alert_df: pd.DataFrame, path: str = "reports/alerts.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    records = alert_df.copy()
    # Serialise timestamps
    for col in records.select_dtypes(include=["datetime64", "datetimetz"]).columns:
        records[col] = records[col].astype(str)
    records.to_json(path, orient="records", indent=2)
    print(f"Alerts saved → {path}  ({len(records)} rows)")


def print_alert_table(alert_df: pd.DataFrame, max_rows: int = 20):
    """Print a compact summary table to stdout."""
    if alert_df.empty:
        print("No alerts generated.")
        return

    display_cols = ["alert_id", "timestamp", "user", "severity", "risk_score", "top_features"]
    show = alert_df[[c for c in display_cols if c in alert_df.columns]].head(max_rows)

    # Simple ASCII table
    col_widths = {c: max(len(c), show[c].astype(str).str.len().max()) for c in show.columns}
    header = "  ".join(c.ljust(col_widths[c]) for c in show.columns)
    sep = "  ".join("-" * col_widths[c] for c in show.columns)
    print("\n" + "=" * len(sep))
    print("UEBA ALERT SUMMARY")
    print("=" * len(sep))
    print(header)
    print(sep)
    for _, row in show.iterrows():
        print("  ".join(str(row[c]).ljust(col_widths[c]) for c in show.columns))
    print(sep)
    print(f"Total alerts: {len(alert_df)}  |  Showing: {len(show)}")
    print("=" * len(sep) + "\n")


def severity_summary(alert_df: pd.DataFrame) -> dict:
    """Return dict of severity → count."""
    if alert_df.empty:
        return {}
    return alert_df["severity"].value_counts().to_dict()
