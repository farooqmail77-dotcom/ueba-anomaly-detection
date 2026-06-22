"""
detector.py
Core two-stage anomaly detection engine.
Stage 1: Isolation Forest (unsupervised)
Stage 2: XGBoost classifier (supervised, trained on labelled synthetic data)
Scores are combined into a weighted risk score with SHAP explanations.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import xgboost as xgb
import shap
import joblib
from pathlib import Path
from typing import List, Optional

warnings.filterwarnings("ignore", category=UserWarning)

FEATURE_COLS = [
    "hour_prob", "day_prob", "country_prob", "server_prob",
    "session_dur_z", "bytes_z", "hour_of_day", "day_of_week",
    "login_success", "session_duration_s", "bytes_transferred",
    "baseline_failure_rate", "baseline_n_obs",
]


class UEBADetector:
    """
    Two-stage UEBA anomaly detector.

    Parameters
    ----------
    if_contamination : float
        Expected fraction of anomalies for Isolation Forest.
    if_weight : float
        Weight of IF score in combined risk score (default 0.35).
    xgb_weight : float
        Weight of XGBoost score in combined risk score (default 0.65).
    alert_threshold : float
        Risk score above which an event is flagged as an alert.
    model_dir : Path
        Directory for persisted model artefacts.
    """

    def __init__(
        self,
        if_contamination: float = 0.05,
        if_weight: float = 0.35,
        xgb_weight: float = 0.65,
        alert_threshold: float = 0.55,
        model_dir: str = "models",
    ):
        self.if_contamination = if_contamination
        self.if_weight = if_weight
        self.xgb_weight = xgb_weight
        self.alert_threshold = alert_threshold
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self._scaler = StandardScaler()
        self._iforest: Optional[IsolationForest] = None
        self._xgb: Optional[xgb.XGBClassifier] = None
        self._explainer = None
        self._is_trained = False

    def fit(self, feature_df: pd.DataFrame) -> "UEBADetector":
        """Train both models. feature_df must include FEATURE_COLS + is_anomaly."""
        X = feature_df[FEATURE_COLS].copy()
        y = feature_df["is_anomaly"].astype(int)
        X_scaled = self._scaler.fit_transform(X)

        # Stage 1: Isolation Forest
        self._iforest = IsolationForest(
            contamination=self.if_contamination,
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
        self._iforest.fit(X_scaled)

        # Stage 2: XGBoost
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, random_state=42
        )
        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        self._xgb = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric="aucpr",
            random_state=42, n_jobs=-1,
        )
        self._xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        y_prob = self._xgb.predict_proba(X_val)[:, 1]
        print(f"XGB Val ROC-AUC : {roc_auc_score(y_val, y_prob):.4f}")
        print(f"XGB Val PR-AUC  : {average_precision_score(y_val, y_prob):.4f}")
        print(classification_report(y_val, (y_prob >= 0.5).astype(int), zero_division=0))

        self._explainer = shap.TreeExplainer(self._xgb)
        self._is_trained = True
        return self

    def score(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with risk_score, alert flag, and SHAP top-3 features."""
        if not self._is_trained:
            raise RuntimeError("Call fit() before score().")

        X = feature_df[FEATURE_COLS].copy()
        X_scaled = self._scaler.transform(X)

        # IF score: invert so higher = more anomalous
        if_raw = self._iforest.decision_function(X_scaled)
        rng = if_raw.max() - if_raw.min() + 1e-9
        if_score = 1.0 - (if_raw - if_raw.min()) / rng

        xgb_score = self._xgb.predict_proba(X_scaled)[:, 1]
        risk = self.if_weight * if_score + self.xgb_weight * xgb_score
        is_alert = (risk >= self.alert_threshold).astype(int)

        shap_vals = self._explainer.shap_values(X_scaled)
        top_feats = self._top_shap_features(shap_vals, n=3)

        out = feature_df[["user", "timestamp", "is_anomaly", "anomaly_type"]].copy()
        out["if_score"] = if_score.round(4)
        out["xgb_score"] = xgb_score.round(4)
        out["risk_score"] = risk.round(4)
        out["alert"] = is_alert
        out["top_features"] = top_feats
        return out

    @staticmethod
    def _top_shap_features(shap_vals: np.ndarray, n: int = 3) -> List[str]:
        results = []
        for row in shap_vals:
            idxs = np.argsort(np.abs(row))[::-1][:n]
            results.append(", ".join(FEATURE_COLS[i] for i in idxs))
        return results

    def save(self):
        joblib.dump(self._scaler, self.model_dir / "scaler.pkl")
        joblib.dump(self._iforest, self.model_dir / "iforest.pkl")
        self._xgb.save_model(str(self.model_dir / "xgb.json"))
        print(f"Models saved to {self.model_dir}/")

    def load(self):
        self._scaler = joblib.load(self.model_dir / "scaler.pkl")
        self._iforest = joblib.load(self.model_dir / "iforest.pkl")
        self._xgb = xgb.XGBClassifier()
        self._xgb.load_model(str(self.model_dir / "xgb.json"))
        self._explainer = shap.TreeExplainer(self._xgb)
        self._is_trained = True
        return self
