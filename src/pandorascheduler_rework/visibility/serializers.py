"""File-writing helpers for visibility artifacts."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def write_csv(path: Path, dataframe: pd.DataFrame) -> None:
    """Persist a DataFrame with legacy-compatible formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    # Placeholder; final implementation will control rounding and dtype casting.
    raise NotImplementedError
