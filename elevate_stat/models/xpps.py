import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ZONE_COL = "SHOT_ZONE_BASIC"
NUMERIC_FEATURES = ["SHOT_DISTANCE", "LOC_X", "LOC_Y", "is_three"]


def _is_three(df):
    return df["SHOT_TYPE"].astype(str).str.contains("3PT").astype(int)


def build_features(shots_df):
    """Returns (X, y). y is None if SHOT_MADE_FLAG is absent."""
    df = shots_df.copy()
    df["is_three"] = _is_three(df)
    zones = pd.get_dummies(df[ZONE_COL].astype(str), prefix="zone").astype(int)
    X = pd.concat([
        df[NUMERIC_FEATURES].astype(float).reset_index(drop=True),
        zones.reset_index(drop=True),
    ], axis=1)
    y = df["SHOT_MADE_FLAG"].astype(int).to_numpy() if "SHOT_MADE_FLAG" in df.columns else None
    return X, y


def fit(shots_df):
    X, y = build_features(shots_df)
    model = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    model.fit(X, y)
    model._feature_columns = list(X.columns)  # remember for apply-time alignment
    return model


def expected_points(model, shots_df):
    """Expected points per shot = P(make) * point_value (3 for a three, else 2)."""
    X, _ = build_features(shots_df)
    X = X.reindex(columns=model._feature_columns, fill_value=0)
    p_make = model.predict_proba(X)[:, 1]
    point_value = np.where(X["is_three"].to_numpy() == 1, 3.0, 2.0)
    return pd.Series(p_make * point_value, index=shots_df.index, name="xpps")


def player_shot_metrics(shots_df, xpps):
    """Per-player: shot-MAKING (points_above_expected) and shot-SELECTION (avg_xpps)."""
    df = shots_df.copy()
    is_three = _is_three(df)
    df["actual_pts"] = df["SHOT_MADE_FLAG"].astype(int) * np.where(is_three == 1, 3, 2)
    df["xpps"] = np.asarray(xpps)
    g = df.groupby("PLAYER_ID")
    out = pd.DataFrame({
        "shots": g.size(),
        "actual_pts": g["actual_pts"].sum(),
        "expected_pts": g["xpps"].sum(),
        "avg_xpps": g["xpps"].mean(),
    })
    out["points_above_expected"] = out["actual_pts"] - out["expected_pts"]
    return out.reset_index()
