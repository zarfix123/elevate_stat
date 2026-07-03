from elevate_stat import run_ingest


def test_parse_args_defaults_to_all_seasons():
    args = run_ingest.parse_args([])
    assert args.seasons is None


def test_parse_args_accepts_season_subset():
    args = run_ingest.parse_args(["--seasons", "2015-16", "2016-17"])
    assert args.seasons == ["2015-16", "2016-17"]


def test_ingest_season_calls_each_stage_in_order(monkeypatch):
    order = []
    monkeypatch.setattr(run_ingest.games, "fetch_games", lambda s: order.append("games"))
    monkeypatch.setattr(run_ingest.games, "game_ids", lambda s: ["001"])
    monkeypatch.setattr(run_ingest.aggregates, "fetch_player_season", lambda s: order.append("player_season"))
    monkeypatch.setattr(run_ingest.shots, "fetch_shots", lambda s: order.append("shots"))
    monkeypatch.setattr(run_ingest.tracking_shots, "fetch_tracking_shots", lambda s: order.append("tracking_shots"))
    monkeypatch.setattr(run_ingest.play_by_play, "fetch_play_by_play", lambda s, gids: order.append("pbp"))
    monkeypatch.setattr(run_ingest.aggregates, "fetch_synergy", lambda s: order.append("synergy"))
    monkeypatch.setattr(run_ingest.aggregates, "fetch_lineups", lambda s: order.append("lineups"))

    run_ingest.ingest_season("2015-16")

    assert order == ["games", "player_season", "shots", "tracking_shots", "pbp", "synergy", "lineups"]
