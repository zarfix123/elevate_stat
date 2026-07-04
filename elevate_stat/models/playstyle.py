import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

PLAY_TYPES = ["Isolation", "Transition", "PRBallHandler", "PRRollman", "Postup",
              "Spotup", "Handoff", "Cut", "OffScreen", "OffRebound", "Misc"]
ADVANCED_FEATURES = ["USG_PCT", "AST_PCT", "OREB_PCT", "DREB_PCT", "TS_PCT", "PACE"]


def build_player_features(advanced_df, synergy_df=None, shots_df=None, min_minutes=500):
    """One row per PLAYER_ID of style features: advanced stats (+ optional synergy
    playtype frequencies + three-point shot share). Indexed by PLAYER_ID, no NaNs."""
    adv = advanced_df.copy()
    if "MIN" in adv.columns:
        # LeagueDashPlayerStats returns per-game minutes; total = MIN * GP.
        gp = adv["GP"] if "GP" in adv.columns else 1
        adv = adv[adv["MIN"] * gp >= min_minutes]
    cols = [c for c in ADVANCED_FEATURES if c in adv.columns]
    feats = adv[["PLAYER_ID"] + cols].set_index("PLAYER_ID")

    if synergy_df is not None and not synergy_df.empty:
        syn = (synergy_df.pivot_table(index="PLAYER_ID", columns="PLAY_TYPE",
                                      values="POSS_PCT", aggfunc="mean")
               .reindex(columns=PLAY_TYPES).add_prefix("pt_"))
        feats = feats.join(syn, how="left")

    if shots_df is not None and not shots_df.empty:
        s = shots_df.copy()
        s["is_three"] = s["SHOT_TYPE"].astype(str).str.contains("3PT").astype(int)
        share = s.groupby("PLAYER_ID")["is_three"].mean().rename("shot_share_three")
        feats = feats.join(share, how="left")

    return feats.fillna(0.0)


def fit(features_df, n_components=8, random_state=0):
    scaler = StandardScaler()
    X = scaler.fit_transform(features_df.to_numpy())
    gmm = GaussianMixture(n_components=n_components, covariance_type="full",
                          random_state=random_state)
    gmm.fit(X)
    return scaler, gmm


def assign(scaler, gmm, features_df):
    """Per player: soft membership per archetype + the top_archetype id."""
    X = scaler.transform(features_df.to_numpy())
    probs = gmm.predict_proba(X)
    out = pd.DataFrame(probs, index=features_df.index,
                       columns=[f"arch_{i}" for i in range(probs.shape[1])])
    out["top_archetype"] = probs.argmax(axis=1)
    return out.reset_index()
