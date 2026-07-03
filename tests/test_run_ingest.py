from elevate_stat import run_ingest


def test_parse_args_defaults_to_all_seasons():
    args = run_ingest.parse_args([])
    assert args.seasons is None


def test_parse_args_accepts_season_subset():
    args = run_ingest.parse_args(["--seasons", "2015-16", "2016-17"])
    assert args.seasons == ["2015-16", "2016-17"]


def _patch_all(monkeypatch, order, failing_stage=None):
    def make(name):
        def stage(*_args, **_kwargs):
            if name == failing_stage:
                raise RuntimeError("boom")
            order.append(name)
        return stage

    monkeypatch.setattr(run_ingest.games, "fetch_games", make("games"))
    monkeypatch.setattr(run_ingest.games, "game_ids", lambda s: ["001"])
    monkeypatch.setattr(run_ingest.aggregates, "fetch_player_season", make("player_season"))
    monkeypatch.setattr(run_ingest.shots, "fetch_shots", make("shots"))
    monkeypatch.setattr(run_ingest.tracking_shots, "fetch_tracking_shots", make("tracking_shots"))
    monkeypatch.setattr(run_ingest.play_by_play, "fetch_play_by_play", make("pbp"))
    monkeypatch.setattr(run_ingest.aggregates, "fetch_synergy", make("synergy"))
    monkeypatch.setattr(run_ingest.aggregates, "fetch_lineups", make("lineups"))


def test_ingest_season_calls_each_stage_in_order(monkeypatch):
    order = []
    _patch_all(monkeypatch, order)
    run_ingest.ingest_season("2015-16")
    assert order == ["games", "player_season", "shots", "tracking_shots", "pbp", "synergy", "lineups"]


def test_a_failing_stage_does_not_abort_the_season(monkeypatch):
    order = []
    _patch_all(monkeypatch, order, failing_stage="player_season")
    run_ingest.ingest_season("2015-16")
    # player_season blew up, but every other stage still ran
    assert order == ["games", "shots", "tracking_shots", "pbp", "synergy", "lineups"]
