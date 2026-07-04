import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PERIOD_SECONDS = 720  # 12-minute regulation quarters
_CLOCK_RE = re.compile(r"PT(?:(\d+)M)?(\d+(?:\.\d+)?)S")


def parse_clock(clock) -> float:
    """'PT10M25.00S' -> 625.0 seconds remaining in the current period."""
    if clock is None or (isinstance(clock, float) and np.isnan(clock)):
        return 0.0
    m = _CLOCK_RE.fullmatch(str(clock).strip())
    if not m:
        return 0.0
    return int(m.group(1) or 0) * 60 + float(m.group(2))


def seconds_remaining(period, clock_seconds) -> float:
    """Game-level seconds remaining. Regulation counts down through Q4;
    overtime (period>=5) uses the remaining OT clock only (approximation)."""
    period = int(period)
    if period <= 4:
        return (4 - period) * PERIOD_SECONDS + clock_seconds
    return clock_seconds


def build_training_frame(game_pbp: pd.DataFrame) -> pd.DataFrame:
    """One game's play-by-play -> rows of (score_diff, seconds_remaining, home_won)."""
    home = pd.to_numeric(game_pbp["scoreHome"], errors="coerce").ffill().fillna(0)
    away = pd.to_numeric(game_pbp["scoreAway"], errors="coerce").ffill().fillna(0)
    secs = [seconds_remaining(p, parse_clock(c)) for p, c in zip(game_pbp["period"], game_pbp["clock"])]
    home_won = 1 if (home.iloc[-1] - away.iloc[-1]) > 0 else 0
    return pd.DataFrame({
        "score_diff": (home - away).to_numpy(dtype=float),
        "seconds_remaining": np.asarray(secs, dtype=float),
        "home_won": home_won,
    })


def _features(score_diff, seconds_remaining):
    score_diff = np.asarray(score_diff, dtype=float)
    seconds_remaining = np.asarray(seconds_remaining, dtype=float)
    sqrt_time = np.sqrt(np.clip(seconds_remaining, 0, None))
    urgency = score_diff / np.sqrt(seconds_remaining + 1.0)
    return np.column_stack([score_diff, sqrt_time, urgency])


def fit(train_df: pd.DataFrame) -> LogisticRegression:
    X = _features(train_df["score_diff"], train_df["seconds_remaining"])
    model = LogisticRegression(max_iter=1000)
    model.fit(X, train_df["home_won"].to_numpy())
    return model


def win_probability(model, score_diff, seconds_remaining):
    X = _features(np.atleast_1d(score_diff), np.atleast_1d(seconds_remaining))
    p = model.predict_proba(X)[:, 1]
    return float(p[0]) if np.ndim(score_diff) == 0 else p


def leverage(model, score_diff, seconds_remaining):
    """Win-prob swing across a ~one-possession (±2 pt) margin change at this state."""
    up = win_probability(model, np.add(score_diff, 2), seconds_remaining)
    dn = win_probability(model, np.subtract(score_diff, 2), seconds_remaining)
    return abs(up - dn) if np.ndim(score_diff) == 0 else np.abs(up - dn)
