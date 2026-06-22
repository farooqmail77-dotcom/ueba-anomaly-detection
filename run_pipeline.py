#!/usr/bin/env python3
"""
run_pipeline.py
End-to-end UEBA pipeline runner.

Usage
-----
  python run_pipeline.py                        # quick demo with synthetic data
  python run_pipeline.py --days 60 --seed 99   # larger dataset
  python run_pipeline.py --log-file path/to/your/auth.csv  # real CSV

CSV must have columns: timestamp, user, source_host, dest_server, country,
hour_of_day, day_of_week, login_success, session_duration_s, bytes_transferred
(is_anomaly and anomaly_type are optional; used for evaluation only).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure repo root is on path when run directly
sys.path.insert(0, str(Path(__file__).parent))

from ueba.data_generator import generate_auth_logs
from ueba.baseline import BaselineBuilder
from ueba.detector import UEBADetector
from ueba.alerts import (
    build_alert_records,
    save_alerts_csv,
    save_alerts_json,
    print_alert_table,
    severity_summary,
)


def parse_args():
    p = argparse.ArgumentParser(description="UEBA anomaly detection pipeline")
    p.add_argument("--days", type=int, default=30, help="Days of synthetic data (default 30)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--train-split", type=float, default=0.7, help="Training fraction")
    p.add_argument("--alert-threshold", type=float, default=0.55)
    p.add_argument("--log-file", type=str, default=None, help="Path to existing CSV log")
    p.add_argument("--out-dir", type=str, default="reports", help="Output directory")
    p.add_argument("--save-models", action="store_true", help="Persist trained models to disk")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    if args.log_file:
        print(f"Loading logs from {args.log_file} ...")
        df = pd.read_csv(args.log_file, parse_dates=["timestamp"])
        # Derive hour/day if not present
        if "hour_of_day" not in df.columns:
            df["hour_of_day"] = df["timestamp"].dt.hour
        if "day_of_week" not in df.columns:
            df["day_of_week"] = df["timestamp"].dt.dayofweek
        # Placeholder ground truth if not present
        if "is_anomaly" not in df.columns:
            df["is_anomaly"] = 0
        if "anomaly_type" not in df.columns:
            df["anomaly_type"] = "unknown"
    else:
        print(f"Generating {args.days} days of synthetic auth logs (seed={args.seed}) ...")
        df = generate_auth_logs(n_days=args.days, seed=args.seed)

    print(f"Total events : {len(df):,}")
    print(f"Anomaly rate : {df['is_anomaly'].mean()*100:.2f}%")

    # ── 2. Train / test split ─────────────────────────────────────────────────
    split_idx = int(len(df) * args.train_split)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    print(f"Train events : {len(train_df):,}  |  Test events : {len(test_df):,}")

    # ── 3. Build per-entity baselines ─────────────────────────────────────────
    print("\nBuilding entity baselines from training data ...")
    builder = BaselineBuilder()
    baselines = builder.fit(train_df)
    print(f"Baselines computed for {len(baselines)} entities.")

    # ── 4. Transform to feature vectors ──────────────────────────────────────
    print("Transforming training data to feature vectors ...")
    train_features = builder.transform(train_df, baselines)
    print("Transforming test data to feature vectors ...")
    test_features = builder.transform(test_df, baselines)

    # ── 5. Train UEBA detector ────────────────────────────────────────────────
    print("\nTraining anomaly detection models ...")
    detector = UEBADetector(alert_threshold=args.alert_threshold)
    detector.fit(train_features)

    if args.save_models:
        detector.save()

    # ── 6. Score test set ─────────────────────────────────────────────────────
    print("\nScoring test events ...")
    scored = detector.score(test_features)

    # ── 7. Build alert records ────────────────────────────────────────────────
    alerts = build_alert_records(scored)
    print_alert_table(alerts)

    sev = severity_summary(alerts)
    print("Severity breakdown:", sev)

    # ── 8. Save outputs ───────────────────────────────────────────────────────
    save_alerts_csv(alerts, str(out_dir / "alerts.csv"))
    save_alerts_json(alerts, str(out_dir / "alerts.json"))

    # Evaluation metrics (only meaningful with labelled data)
    if "is_anomaly" in scored.columns and scored["is_anomaly"].sum() > 0:
        from sklearn.metrics import (
            classification_report, roc_auc_score, average_precision_score
        )
        y_true = scored["is_anomaly"]
        y_pred = scored["alert"]
        y_prob = scored["risk_score"]
        print("\n── Test-set evaluation ──────────────────────────────────")
        print(classification_report(y_true, y_pred, zero_division=0))
        try:
            print(f"ROC-AUC : {roc_auc_score(y_true, y_prob):.4f}")
            print(f"PR-AUC  : {average_precision_score(y_true, y_prob):.4f}")
        except Exception:
            pass

    print("\nDone. Reports written to:", out_dir.resolve())


if __name__ == "__main__":
    main()
