import re
from elevate_stat import config


def test_seasons_span_2015_16_through_2025_26():
    assert config.SEASONS[0] == "2015-16"
    assert config.SEASONS[-1] == "2025-26"
    assert len(config.SEASONS) == 11


def test_every_season_matches_nba_format():
    for season in config.SEASONS:
        assert re.fullmatch(r"\d{4}-\d{2}", season), season


def test_tuning_constants_present():
    assert config.REQUEST_DELAY > 0
    assert config.MAX_RETRIES >= 1
    assert config.TIMEOUT >= 30
    assert config.SEASON_TYPES == ["Regular Season", "Playoffs"]
