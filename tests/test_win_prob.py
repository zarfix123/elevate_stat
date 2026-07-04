import numpy as np
import pandas as pd
from elevate_stat.models import win_prob


def test_parse_clock():
    assert win_prob.parse_clock("PT10M25.00S") == 625.0
    assert win_prob.parse_clock("PT0M5.00S") == 5.0


def test_seconds_remaining_counts_down_through_game():
    assert win_prob.seconds_remaining(1, 720) == 2880  # start of Q1: 48 min
    assert win_prob.seconds_remaining(2, 360) == 1800
    assert win_prob.seconds_remaining(4, 0) == 0


def test_build_training_frame_uses_final_score_for_label():
    g = pd.DataFrame({
        "scoreHome": [0, 2, 4, 10],
        "scoreAway": [0, 3, 4, 8],
        "period": [1, 1, 2, 4],
        "clock": ["PT12M0.0S", "PT10M0.0S", "PT6M0.0S", "PT0M0.0S"],
    })
    tf = win_prob.build_training_frame(g)
    assert (tf["home_won"] == 1).all()          # final 10 > 8
    assert tf["score_diff"].iloc[-1] == 2.0
    assert tf["seconds_remaining"].iloc[0] == 2880


def _synthetic_training():
    rng = np.random.RandomState(0)
    rows = []
    for _ in range(600):
        diff = rng.randint(-20, 21)
        secs = rng.randint(1, 2880)
        prob = 1.0 / (1.0 + np.exp(-(diff / (1.0 + secs / 600.0))))  # lead matters more late
        rows.append((diff, secs, int(rng.rand() < prob)))
    return pd.DataFrame(rows, columns=["score_diff", "seconds_remaining", "home_won"])


def test_win_probability_increases_with_lead():
    m = win_prob.fit(_synthetic_training())
    assert win_prob.win_probability(m, 15, 120) > win_prob.win_probability(m, -15, 120)


def test_win_probability_near_half_at_tie_early():
    m = win_prob.fit(_synthetic_training())
    assert 0.35 < win_prob.win_probability(m, 0, 2500) < 0.65


def test_leverage_higher_when_close_and_late():
    m = win_prob.fit(_synthetic_training())
    assert win_prob.leverage(m, 0, 30) > win_prob.leverage(m, 25, 30)     # vs blowout
    assert win_prob.leverage(m, 0, 30) > win_prob.leverage(m, 0, 2500)    # vs early
