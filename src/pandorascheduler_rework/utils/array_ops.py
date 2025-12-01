"""Array processing utilities for observation scheduling."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np


def remove_short_sequences(
    array: np.ndarray, 
    sequence_too_short: int
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Remove visibility sequences shorter than threshold.
    
    Args:
        array: Binary visibility array (1=visible, 0=not visible)
        sequence_too_short: Minimum sequence length to keep
        
    Returns:
        Tuple of (cleaned array, list of removed spans)
    """
    cleaned = np.asarray(array, dtype=float).copy()
    start_index = None
    spans: List[Tuple[int, int]] = []

    for idx, value in enumerate(cleaned):
        if value == 1 and start_index is None:
            start_index = idx
            continue
        if value == 0 and start_index is not None:
            if idx - start_index < sequence_too_short:
                spans.append((start_index, idx - 1))
            start_index = None

    if start_index is not None and len(cleaned) - start_index < sequence_too_short:
        spans.append((start_index, len(cleaned) - 1))

    for start_idx, stop_idx in spans:
        cleaned[start_idx : stop_idx + 1] = 0.0

    return cleaned, spans


def break_long_sequences(
    start: datetime, 
    end: datetime, 
    step: timedelta
) -> List[Tuple[datetime, datetime]]:
    """Break long time range into smaller chunks.
    
    Args:
        start: Start time
        end: End time
        step: Maximum chunk duration
        
    Returns:
        List of (start, end) tuples for each chunk
    """
    ranges: List[Tuple[datetime, datetime]] = []
    current = start
    while current < end:
        next_val = min(current + step, end)
        ranges.append((current, next_val))
        current = next_val
    return ranges
