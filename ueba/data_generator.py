"""
data_generator.py
Generates synthetic authentication/access logs for UEBA testing.
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

USERS = [f"user_{i:03d}" for i in range(1, 51)]
HOSTS = [f"host_{i:02d}" for i in range(1, 21)]
SERVERS = [f"srv_{s}" for s in ["auth", "files", "db", "hr", "finance", "it"]]
COUNTRIES = ["US", "US", "US", "GB", "DE", "CA", "FR", "US", "US", "CN", "RU"]
WORK_HOURS = range(7, 19)
ANOMALY_RATE = 0.04


def _random_ts(base: datetime, jitter_hours: float = 1.0) -> datetime:
    offset = random.gauss(0, jitter_hours * 3600)
    return base + timedelta(seconds=offset)


def generate_auth_logs(
    n_days: int = 30,
    seed: int = 42,
    anomaly_rate: float = ANOMALY_RATE,
) -> pd.DataFrame:
    """
    Return a DataFrame of synthetic authentication events.

    Columns
    -------
    timestamp, user, source_host, dest_server, country,
    hour_of_day, day_of_week, login_success, session_duration_s,
    bytes_transferred, is_anomaly, anomaly_type
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    records = []
    start = datetime(2024, 1, 1, 0, 0, 0)

    for day in range(n_days):
        day_start = start + timedelta(days=day)
        day_of_week = day_start.weekday()

        for user in USERS:
            if day_of_week < 5:
                n_sessions = rng.randint(1, 4)
            else:
                n_sessions = rng.randint(0, 1)

            for _ in range(n_sessions):
                hour = rng.choice(list(WORK_HOURS))
                ts = day_start.replace(hour=hour) + timedelta(minutes=rng.randint(0, 59))

                is_anomaly = rng.random() < anomaly_rate
                anomaly_type = "normal"

                if is_anomaly:
                    atype = rng.choice([
                        "off_hours_login",
                        "unusual_country",
                        "high_byte_transfer",
                        "brute_force_failure",
                        "lateral_movement",
                    ])
                    anomaly_type = atype
                    if atype == "off_hours_login":
                        hour = rng.choice([0, 1, 2, 3, 4, 22, 23])
                        ts = day_start.replace(hour=hour) + timedelta(minutes=rng.randint(0, 59))
                        country = rng.choice(["US", "GB"])
                    elif atype == "unusual_country":
                        country = rng.choice(["CN", "RU", "KP", "IR"])
                    elif atype == "high_byte_transfer":
                        country = rng.choice(COUNTRIES)
                    elif atype == "brute_force_failure":
                        country = rng.choice(COUNTRIES)
                    else:
                        country = "US"
                else:
                    country = rng.choice(COUNTRIES[:9])

                login_success = True if anomaly_type != "brute_force_failure" else (rng.random() < 0.2)
                session_dur = int(np_rng.lognormal(7, 1)) if login_success else rng.randint(1, 30)

                if anomaly_type == "high_byte_transfer":
                    bytes_xfer = int(np_rng.lognormal(18, 1))
                elif anomaly_type == "lateral_movement":
                    bytes_xfer = int(np_rng.lognormal(12, 0.5))
                else:
                    bytes_xfer = int(np_rng.lognormal(10, 1.5))

                dest = rng.choice(SERVERS)
                if anomaly_type == "lateral_movement":
                    dest = rng.choice(["srv_hr", "srv_finance", "srv_db"])

                records.append({
                    "timestamp": ts,
                    "user": user,
                    "source_host": rng.choice(HOSTS),
                    "dest_server": dest,
                    "country": country,
                    "hour_of_day": ts.hour,
                    "day_of_week": ts.weekday(),
                    "login_success": int(login_success),
                    "session_duration_s": session_dur,
                    "bytes_transferred": bytes_xfer,
                    "is_anomaly": int(is_anomaly),
                    "anomaly_type": anomaly_type,
                })

    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_auth_logs(n_days=30)
    print(f"Generated {len(df):,} auth events")
    print(df.head())
    print("\nAnomaly breakdown:")
    print(df["anomaly_type"].value_counts())
    df.to_csv("data/auth_logs.csv", index=False)
    print("\nSaved to data/auth_logs.csv")
