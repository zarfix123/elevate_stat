import time
from typing import Callable
from elevate_stat import config


def call(
    endpoint_factory: Callable,
    *,
    delay: float | None = None,
    max_retries: int | None = None,
    timeout: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs,
):
    """Call an nba_api endpoint with a polite delay + retry/backoff.

    `endpoint_factory(**kwargs)` returns an object with `.get_data_frames()`.
    Real usage passes an nba_api endpoint class (e.g. PlayByPlayV2).
    """
    delay = config.REQUEST_DELAY if delay is None else delay
    max_retries = config.MAX_RETRIES if max_retries is None else max_retries
    timeout = config.TIMEOUT if timeout is None else timeout

    last_err = None
    for attempt in range(max_retries):
        sleep(delay)  # be polite before every attempt
        try:
            endpoint = endpoint_factory(timeout=timeout, **kwargs)
            return endpoint.get_data_frames()
        except Exception as err:  # noqa: BLE001 — network layer is genuinely unpredictable
            last_err = err
            backoff = min(2 ** attempt, 60)
            sleep(backoff)
    raise RuntimeError(f"call failed after {max_retries} attempts: {last_err}") from last_err
