import pytest
from elevate_stat.client import call


class _FakeEndpoint:
    """Fails `fail_times` then returns a sentinel from get_data_frames()."""
    instances = 0

    def __init__(self, fail_times=0, **kwargs):
        type(self).instances += 1
        self._fail_times = fail_times
        self.kwargs = kwargs

    def get_data_frames(self):
        if type(self).instances <= self._fail_times:
            raise ConnectionError("boom")
        return ["OK"]


def _make_fake(fail_times):
    _FakeEndpoint.instances = 0

    def factory(**kwargs):
        return _FakeEndpoint(fail_times=fail_times, **kwargs)

    return factory


def test_call_returns_data_frames_on_success():
    result = call(_make_fake(0), sleep=lambda _: None, game_id="X")
    assert result == ["OK"]


def test_call_retries_then_succeeds():
    result = call(_make_fake(2), sleep=lambda _: None, max_retries=5)
    assert result == ["OK"]
    assert _FakeEndpoint.instances == 3  # 2 failures + 1 success


def test_call_raises_after_exhausting_retries():
    with pytest.raises(RuntimeError):
        call(_make_fake(99), sleep=lambda _: None, max_retries=3)
