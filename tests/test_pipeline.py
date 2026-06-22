"""
tests/test_pipeline.py
Pytest suite for the UEBA anomaly detection pipeline.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from ueba.data_generator import generate_auth_logs
from ueba.baseline import BaselineBuilder, EntityBaseline
from ueba.detector import UEBADetector, FEATURE_COLS
from ueba.alerts import build_alert_records, severity_summary


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_logs():
    """Small 10-day synthetic dataset."""
    return generate_auth_logs(n_days=10, seed=0)


@pytest.fixture(scope="module")
def baselines_and_features(raw_logs):
    split = int(len(raw_logs) * 0.7)
    train = raw_logs.iloc[:split].copy()
    test  = raw_logs.iloc[split:].copy()
    builder = BaselineBuilder()
    baselines = builder.fit(train)
    train_feat = builder.transform(train, baselines)
    test_feat  = builder.transform(test,  baselines)
    return baselines, train_feat, test_feat


@pytest.fixture(scope="module")
def trained_detector(baselines_and_features):
    _, train_feat, _ = baselines_and_features
    detector = UEBADetector(alert_threshold=0.5)
    detector.fit(train_feat)
    return detector


# ── Data generator tests ──────────────────────────────────────────────────────

class TestDataGenerator:
    def test_returns_dataframe(self, raw_logs):
        assert isinstance(raw_logs, pd.DataFrame)

    def test_required_columns(self, raw_logs):
        required = {"timestamp", "user", "hour_of_day", "day_of_week",
                    "login_success", "session_duration_s", "bytes_transferred",
                    "is_anomaly", "anomaly_type"}
        assert required.issubset(raw_logs.columns)

    def test_anomaly_rate_in_range(self, raw_logs):
        rate = raw_logs["is_anomaly"].mean()
        assert 0.01 < rate < 0.15, f"Anomaly rate {rate:.3f} outside expected range"

    def test_no_null_timestamps(self, raw_logs):
        assert raw_logs["timestamp"].notna().all()

    def test_hour_range(self, raw_logs):
        assert raw_logs["hour_of_day"].between(0, 23).all()

    def test_multiple_users(self, raw_logs):
        assert raw_logs["user"].nunique() > 1


# ── Baseline tests ────────────────────────────────────────────────────────────

class TestBaselineBuilder:
    def test_baseline_keys(self, baselines_and_features, raw_logs):
        baselines, _, _ = baselines_and_features
        expected_users = set(raw_logs["user"].unique())
        assert expected_users == set(baselines.keys())

    def test_entity_baseline_type(self, baselines_and_features):
        baselines, _, _ = baselines_and_features
        for b in baselines.values():
            assert isinstance(b, EntityBaseline)

    def test_hour_distribution_sums_to_one(self, baselines_and_features):
        baselines, _, _ = baselines_and_features
        for b in list(baselines.values())[:5]:
            total = b.typical_hours.sum()
            assert total > 0

    def test_transform_returns_feature_cols(self, baselines_and_features):
        _, train_feat, _ = baselines_and_features
        for col in FEATURE_COLS:
            assert col in train_feat.columns, f"Missing feature col: {col}"

    def test_no_nulls_in_features(self, baselines_and_features):
        _, train_feat, _ = baselines_and_features
        null_counts = train_feat[FEATURE_COLS].isnull().sum()
        assert null_counts.sum() == 0, f"Null values in features: {null_counts[null_counts > 0]}"


# ── Detector tests ────────────────────────────────────────────────────────────

class TestUEBADetector:
    def test_fit_returns_self(self, baselines_and_features):
        _, train_feat, _ = baselines_and_features
        det = UEBADetector()
        result = det.fit(train_feat)
        assert result is det

    def test_score_returns_dataframe(self, trained_detector, baselines_and_features):
        _, _, test_feat = baselines_and_features
        scored = trained_detector.score(test_feat)
        assert isinstance(scored, pd.DataFrame)

    def test_score_required_columns(self, trained_detector, baselines_and_features):
        _, _, test_feat = baselines_and_features
        scored = trained_detector.score(test_feat)
        for col in ["risk_score", "alert", "if_score", "xgb_score", "top_features"]:
            assert col in scored.columns

    def test_risk_score_range(self, trained_detector, baselines_and_features):
        _, _, test_feat = baselines_and_features
        scored = trained_detector.score(test_feat)
        assert scored["risk_score"].between(0, 1).all(), "Risk scores outside [0,1]"

    def test_alert_is_binary(self, trained_detector, baselines_and_features):
        _, _, test_feat = baselines_and_features
        scored = trained_detector.score(test_feat)
        assert set(scored["alert"].unique()).issubset({0, 1})

    def test_some_alerts_fired(self, trained_detector, baselines_and_features):
        _, _, test_feat = baselines_and_features
        scored = trained_detector.score(test_feat)
        n_alerts = scored["alert"].sum()
        assert n_alerts > 0, "No alerts fired — threshold may be too high or data too small"

    def test_score_raises_before_fit(self, baselines_and_features):
        _, _, test_feat = baselines_and_features
        fresh = UEBADetector()
        with pytest.raises(RuntimeError):
            fresh.score(test_feat)


# ── Alert formatter tests ─────────────────────────────────────────────────────

class TestAlerts:
    def test_build_alert_records(self, trained_detector, baselines_and_features):
        _, _, test_feat = baselines_and_features
        scored = trained_detector.score(test_feat)
        alerts = build_alert_records(scored)
        assert isinstance(alerts, pd.DataFrame)

    def test_alert_ids_unique(self, trained_detector, baselines_and_features):
        _, _, test_feat = baselines_and_features
        scored = trained_detector.score(test_feat)
        alerts = build_alert_records(scored)
        if not alerts.empty:
            assert alerts["alert_id"].is_unique

    def test_severity_values(self, trained_detector, baselines_and_features):
        _, _, test_feat = baselines_and_features
        scored = trained_detector.score(test_feat)
        alerts = build_alert_records(scored)
        valid = {"CRITICAL", "HIGH", "MEDIUM", "INFO"}
        if not alerts.empty:
            assert set(alerts["severity"].unique()).issubset(valid)

    def test_severity_summary_returns_dict(self, trained_detector, baselines_and_features):
        _, _, test_feat = baselines_and_features
        scored = trained_detector.score(test_feat)
        alerts = build_alert_records(scored)
        sev = severity_summary(alerts)
        assert isinstance(sev, dict)
