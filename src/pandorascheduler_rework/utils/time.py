"""Time and datetime utility functions for the Pandora scheduler."""

from __future__ import annotations

from datetime import datetime, timedelta


def round_to_nearest_second(timestamp: datetime) -> datetime:
    """Return ``timestamp`` rounded to the nearest whole second.
    
    Parameters
    ----------
    timestamp
        Datetime to round.
        
    Returns
    -------
    datetime
        Input timestamp rounded to the nearest whole second, with microseconds set to zero.
        
    Examples
    --------
    >>> from datetime import datetime
    >>> round_to_nearest_second(datetime(2026, 1, 1, 12, 0, 0, 499_999))
    datetime.datetime(2026, 1, 1, 12, 0, 0)
    >>> round_to_nearest_second(datetime(2026, 1, 1, 12, 0, 0, 500_000))
    datetime.datetime(2026, 1, 1, 12, 0, 1)
    """
    if timestamp.microsecond >= 500_000:
        timestamp = timestamp + timedelta(seconds=1)
    return timestamp.replace(microsecond=0)


__all__ = ["round_to_nearest_second"]
