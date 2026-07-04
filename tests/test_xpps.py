import numpy as np
import pandas as pd
from elevate_stat.models import xpps


def test_build_features_has_is_three_and_target():
    df = pd.DataFrame({
        "SHOT_DISTANCE": [2, 25], "LOC_X": [0, 100], "LOC_Y": [10, 200],
        "SHOT_ZONE_BASIC": ["Restricted Area", "Above the Break 3"],
        "SHOT_TYPE": ["2PT Field Goal", "3PT Field Goal"], "SHOT_MADE_FLAG": [1, 0],
    })
    X, y = xpps.build_features(df)
    assert "is_three" in X.columns
    assert X["is_three"].tolist() == [0, 1]
    assert list(y) == [1, 0]


def _synthetic_shots(n=600):
    rng = np.random.RandomState(0)
    dist = rng.choice([2, 26], size=n)
    made = np.where(dist == 2, rng.rand(n) < 0.62, rng.rand(n) < 0.35).astype(int)
    return pd.DataFrame({
        "PLAYER_ID": rng.randint(1, 4, size=n),
        "SHOT_DISTANCE": dist, "LOC_X": 0, "LOC_Y": dist * 10,
        "SHOT_ZONE_BASIC": np.where(dist == 2, "Restricted Area", "Above the Break 3"),
        "SHOT_TYPE": np.where(dist == 2, "2PT Field Goal", "3PT Field Goal"),
        "SHOT_MADE_FLAG": made,
    })


def test_model_gives_higher_make_prob_for_short_shots():
    df = _synthetic_shots()
    m = xpps.fit(df)
    short = df[df["SHOT_DISTANCE"] == 2].head(1)
    long_ = df[df["SHOT_DISTANCE"] == 26].head(1)
    Xs, _ = xpps.build_features(short); Xs = Xs.reindex(columns=m._feature_columns, fill_value=0)
    Xl, _ = xpps.build_features(long_); Xl = Xl.reindex(columns=m._feature_columns, fill_value=0)
    assert m.predict_proba(Xs)[0, 1] > m.predict_proba(Xl)[0, 1]


def test_expected_points_scales_by_shot_value():
    df = _synthetic_shots()
    m = xpps.fit(df)
    xp = xpps.expected_points(m, df)
    X, _ = xpps.build_features(df); X = X.reindex(columns=m._feature_columns, fill_value=0)
    p = m.predict_proba(X)[:, 1]
    pv = np.where(X["is_three"].to_numpy() == 1, 3.0, 2.0)
    assert np.allclose(xp.to_numpy(), p * pv)


def test_player_shot_metrics_aggregates():
    df = pd.DataFrame({
        "PLAYER_ID": [1, 1, 2],
        "SHOT_TYPE": ["2PT Field Goal", "3PT Field Goal", "2PT Field Goal"],
        "SHOT_MADE_FLAG": [1, 1, 0],
    })
    xp = pd.Series([1.0, 1.5, 0.8], index=df.index)
    out = xpps.player_shot_metrics(df, xp).set_index("PLAYER_ID")
    assert out.loc[1, "actual_pts"] == 5            # 2 + 3
    assert abs(out.loc[1, "expected_pts"] - 2.5) < 1e-9
    assert abs(out.loc[1, "points_above_expected"] - 2.5) < 1e-9
    assert out.loc[1, "shots"] == 2
    assert out.loc[2, "actual_pts"] == 0
    assert abs(out.loc[2, "points_above_expected"] + 0.8) < 1e-9
