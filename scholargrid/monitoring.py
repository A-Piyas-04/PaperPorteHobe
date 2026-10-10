"""Optional error tracking (Sentry) for the pipeline and the app.

Enabled only when ``SENTRY_DSN`` is set and ``sentry-sdk`` is installed; a
no-op otherwise. No personal data is sent (``send_default_pii=False``).
"""
from __future__ import annotations

import os

from .utils import get_logger

log = get_logger("monitoring")
_initialised = False


def init_error_tracking(component: str, release: str | None = None) -> bool:
    global _initialised
    dsn = os.environ.get("SENTRY_DSN", "")
    if _initialised or not dsn:
        return _initialised
    try:
        import sentry_sdk  # type: ignore
    except ImportError:
        log.warning("SENTRY_DSN is set but sentry-sdk is not installed; error tracking disabled.")
        return False
    sentry_sdk.init(dsn=dsn, release=release, environment=os.environ.get("SCHOLARGRID_ENV", "development"),
                    send_default_pii=False, traces_sample_rate=float(os.environ.get("SENTRY_TRACES", "0")))
    sentry_sdk.set_tag("component", component)
    _initialised = True
    log.info("Sentry error tracking enabled for %s.", component)
    return True


def capture_exception(exc: BaseException) -> None:
    log.error("Unhandled exception", exc_info=(type(exc), exc, exc.__traceback__))
    if not _initialised:
        return
    import sentry_sdk  # type: ignore

    sentry_sdk.capture_exception(exc)
