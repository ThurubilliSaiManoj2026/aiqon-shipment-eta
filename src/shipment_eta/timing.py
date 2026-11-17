"""UTC timing and dispatch rules, independent of random generation."""

import math
import pandas as pd

NS_PER_HOUR = 3_600_000_000_000


def utc_timestamp(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Timestamp must be present and timezone-aware")
    return stamp.tz_convert("UTC").as_unit("ns")


def hours_between(end, start) -> float:
    return (utc_timestamp(end) - utc_timestamp(start)).value / NS_PER_HOUR


def next_dispatch(ready, first_slot, interval_h: float) -> pd.Timestamp:
    """First allowed slot >= readiness; exact slot boundaries are inclusive.

    Integer nanoseconds avoid floating-point ceil errors at dispatch boundaries.
    """
    if isinstance(interval_h, bool) or not math.isfinite(interval_h) or interval_h <= 0:
        raise ValueError("Dispatch interval must be finite and positive")
    step = pd.Timedelta(hours=interval_h).value
    if step <= 0:
        raise ValueError("Dispatch interval is below timestamp resolution")
    ready, first_slot = utc_timestamp(ready), utc_timestamp(first_slot)
    delta = (ready - first_slot).value
    slot_index = max(0, (delta + step - 1) // step)
    return first_slot + pd.Timedelta(slot_index * step, unit="ns")
