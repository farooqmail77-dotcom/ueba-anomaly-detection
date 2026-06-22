"""
baseline.py
Builds per-entity behavioral baselines from historical auth logs.
Computes rolling statistics used as features by the anomaly detector.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class EntityBaseline:
    """Per-user statistical baseline computed over a training window."""
    user: str
    # Login timing
    typical_hours: np.ndarray = field(default_factory=lambda: np.zeros(24))
    typical_days: np.ndarray = field(default_factory=lambda: np.zeros(7))
    # Country distribution
    country_counts: Dict[str, int] = field(default_factory=dict)
    # Session stats
    session_dur_mean: float = 0.0
    session_dur_std: float = 1.0
    bytes_mean: float = 0.0
    bytes_std: float = 1.0
    # Failure rate
    failure_rate: float = 0.0
    # Server access pattern
    server_counts: Dict[str, int] = field(default_factory=dict)
    # Total observations
    n_obs: int = 0

    def hour_probability(self, hour: int) -> float:
        total = self.typical_hours.sum()
        if total == 0:
            return 1.0 / 24
        return float(self.typical_hours[hour]) / total

    def day_probability(self, day: int) -> float:
        total = self.typical_days.sum()
        if total == 0:
            return 1.0 / 7
        return float(self.typical_days[day]) / total

    def country_probability(self, country: str) -> float:
        total = sum(self.country_counts.values())
        if total == 0:
            return 0.01
        return self.country_counts.get(country, 0) / total

    def server_probability(self, server: str) -> float:
        total = sum(self.server_counts.values())
        if total == 0:
            return 0.01
        return self.server_counts.get(server, 0) / total


class BaselineBuilder:
    """
    Computes per-entity baselines from a training DataFrame.

    Usage
    -----
    builder = BaselineBuilder(lookback_days=21)
    baselines = builder.fit(train_df)
    feature_df = builder.transform(test_df, baselines)
    """

    def __init__(self, lookback_days: int = 21):
        self.lookback_days = lookback_days

    def fit(self, df: pd.DataFrame) -> Dict[str, EntityBaseline]:
        """Compute baselines for all users in df."""
        baselines: Dict[str, EntityBaseline] = {}
        for user, grp in df.groupby("user"):
            b = EntityBaseline(user=str(user))
            # Hour distribution
            for h in grp["hour_of_day"]:
                b.typical_hours[int(h)] += 1
            # Day distribution
            for d in grp["day_of_week"]:
                b.typical_days[int(d)] += 1
            # Country counts
            for c in grp["country"]:
                b.country_counts[str(c)] = b.country_counts.get(str(c), 0) + 1
            # Server counts
            for s in grp["dest_server"]:
                b.server_counts[str(s)] = b.server_counts.get(str(s), 0) + 1
            # Session duration
            dur = grp["session_duration_s"]
            b.session_dur_mean = float(dur.mean())
            b.session_dur_std = max(float(dur.std()), 1.0)
            # Bytes
            byt = grp["bytes_transferred"]
            b.bytes_mean = float(byt.mean())
            b.bytes_std = max(float(byt.std()), 1.0)
            # Failure rate
            b.failure_rate = float(1 - grp["login_success"].mean())
            b.n_obs = len(grp)
            baselines[str(user)] = b
        return baselines

    def transform(
        self,
        df: pd.DataFrame,
        baselines: Dict[str, EntityBaseline],
    ) -> pd.DataFrame:
        """
        Convert raw events to feature vectors using per-entity baselines.

        Each row becomes a Z-score or probability-based deviation feature.
        """
        rows = []
        for _, row in df.iterrows():
            user = str(row["user"])
            b = baselines.get(user)

            if b is None or b.n_obs < 5:
                # Insufficient history — use population-average placeholder
                rows.append(self._cold_start_features(row))
                continue

            hour_prob = b.hour_probability(int(row["hour_of_day"]))
            day_prob = b.day_probability(int(row["day_of_week"]))
            country_prob = b.country_probability(str(row["country"]))
            server_prob = b.server_probability(str(row["dest_server"]))

            dur_z = (float(row["session_duration_s"]) - b.session_dur_mean) / b.session_dur_std
            bytes_z = (float(row["bytes_transferred"]) - b.bytes_mean) / b.bytes_std

            rows.append({
                "user": user,
                "timestamp": row["timestamp"],
                # Deviation features (lower probability = more anomalous)
                "hour_prob": hour_prob,
                "day_prob": day_prob,
                "country_prob": country_prob,
                "server_prob": server_prob,
                # Z-scores
                "session_dur_z": dur_z,
                "bytes_z": bytes_z,
                # Raw features kept for boosting model
                "hour_of_day": int(row["hour_of_day"]),
                "day_of_week": int(row["day_of_week"]),
                "login_success": int(row["login_success"]),
                "session_duration_s": float(row["session_duration_s"]),
                "bytes_transferred": float(row["bytes_transferred"]),
                "baseline_failure_rate": b.failure_rate,
                "baseline_n_obs": b.n_obs,
                # Ground truth (only available for evaluation, not production)
                "is_anomaly": int(row.get("is_anomaly", 0)),
                "anomaly_type": str(row.get("anomaly_type", "normal")),
            })

        return pd.DataFrame(rows)

    @staticmethod
    def _cold_start_features(row) -> dict:
        return {
            "user": str(row["user"]),
            "timestamp": row["timestamp"],
            "hour_prob": 1.0 / 24,
            "day_prob": 1.0 / 7,
            "country_prob": 0.1,
            "server_prob": 0.1,
            "session_dur_z": 0.0,
            "bytes_z": 0.0,
            "hour_of_day": int(row["hour_of_day"]),
            "day_of_week": int(row["day_of_week"]),
            "login_success": int(row["login_success"]),
            "session_duration_s": float(row["session_duration_s"]),
            "bytes_transferred": float(row["bytes_transferred"]),
            "baseline_failure_rate": 0.0,
            "baseline_n_obs": 0,
            "is_anomaly": int(row.get("is_anomaly", 0)),
            "anomaly_type": str(row.get("anomaly_type", "normal")),
        }
