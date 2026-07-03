import logging

log = logging.getLogger("elevate_stat.ingest")


def for_each(items, fn, *, label):
    """Run fn(item) for each item, isolating failures.

    A single failing unit is logged and skipped so one bad item never aborts a
    multi-hour run. Returns the list of (item, exception) that failed; re-running
    the pipeline retries them (skip-if-exists covers everything already saved).
    """
    failures = []
    for item in items:
        try:
            fn(item)
        except Exception as err:  # noqa: BLE001 — isolate any single-unit failure
            log.warning("skip %s=%s — %s: %s", label, item, type(err).__name__, str(err)[:120])
            failures.append((item, err))
    return failures
