import pandas as pd
from elevate_stat import pbp_lineups as pl


def _game(events):
    """events: list of (teamId, personId, playerName, actionType, description)."""
    rows = [{
        "actionNumber": i + 1, "period": 1, "clock": "PT12M00.00S",
        "teamId": t, "personId": p, "playerName": n, "actionType": a, "description": d,
    } for i, (t, p, n, a, d) in enumerate(events)]
    return pd.DataFrame(rows)


def _five(team, ids, action="Made Shot"):
    return [(team, i, f"P{i}", action, "") for i in ids]


def test_parse_sub():
    assert pl.parse_sub("SUB: Niang FOR Allen") == "Niang"
    assert pl.parse_sub("MISS Jones 3PT") is None


def test_reconstruct_tracks_starters_and_a_substitution():
    events = (
        _five(100, [1, 2, 3, 4, 5])
        + _five(200, [11, 12, 13, 14, 15])
        + [(100, 1, "P1", "Substitution", "SUB: P6 FOR P1")]
        + [(100, 6, "P6", "Made Shot", "")]
    )
    out, ok = pl.reconstruct(_game(events))
    assert ok is True
    assert set(out["on_a"].iloc[0]) == {1, 2, 3, 4, 5}      # before sub
    assert set(out["on_a"].iloc[-1]) == {2, 3, 4, 5, 6}     # after sub (P1 out, P6 in)
    assert set(out["on_b"].iloc[0]) == {11, 12, 13, 14, 15}


def test_starter_subbed_out_before_acting_is_still_counted():
    # P5 never records an event but is subbed out -> must be inferred as a starter
    events = (
        _five(100, [1, 2, 3, 4])
        + _five(200, [11, 12, 13, 14, 15])
        + [(100, 5, "P5", "Substitution", "SUB: P6 FOR P5")]
        + [(100, 6, "P6", "Made Shot", "")]
    )
    out, ok = pl.reconstruct(_game(events))
    assert ok is True
    assert 5 in set(out["on_a"].iloc[0])                    # discovered via the sub-out


def test_unresolvable_game_flags_not_ok_and_marks_invalid():
    # home has only 4 discoverable players (P5 never acts, never subs) -> can't reach 5
    events = _five(100, [1, 2, 3, 4]) + _five(200, [11, 12, 13, 14, 15])
    out, ok = pl.reconstruct(_game(events))
    assert ok is False
    assert not out["valid"].any()          # the whole (single) period is invalid


def test_clean_game_is_all_valid():
    events = _five(100, [1, 2, 3, 4, 5]) + _five(200, [11, 12, 13, 14, 15])
    out, ok = pl.reconstruct(_game(events))
    assert ok is True and out["valid"].all()


def test_same_last_name_resolved_by_full_name():
    # two Williams on team 100: Jalen(1) starts, Jaylin(2) subs in for P6
    events = [
        (100, 1, "Williams", "Made Shot", ""),
    ] + _five(100, [3, 4, 5, 6]) + _five(200, [11, 12, 13, 14, 15]) + [
        (100, 6, "P6", "Substitution", "SUB: Jay. Williams FOR P6"),
        (100, 2, "Williams", "Made Shot", ""),
    ]
    id_full = {1: "Jalen Williams", 2: "Jaylin Williams"}
    out, ok = pl.reconstruct(_game(events), id_fullname=id_full)
    assert ok is True
    assert set(out["on_a"].iloc[-1]) == {1, 2, 3, 4, 5}   # Jaylin(2) in for P6, Jalen(1) stays
