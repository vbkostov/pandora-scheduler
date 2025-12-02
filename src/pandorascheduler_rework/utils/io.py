"""Input/Output utilities with caching."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)


def _read_csv_with_mtime(file_path: str, mtime: Optional[float]) -> Optional[pd.DataFrame]:
    """Internal reader cached by (file_path, mtime).

    The wrapper `read_csv_cached` computes a file's mtime and passes
    it here so the LRU cache key includes the mtime. That way, when a
    file is overwritten (mtime changes), this cache entry is invalidated
    and the new contents are read.
    """
    path = Path(file_path)
    try:
        return pd.read_csv(path)
    except Exception as e:
        LOGGER.error(f"Error reading {path}: {e}")
        return None


# cache on (file_path, mtime) -- mtime is a float or None (both hashable)
_read_csv_with_mtime = lru_cache(maxsize=64)(_read_csv_with_mtime)


def read_csv_cached(file_path: str) -> Optional[pd.DataFrame]:
    """Read CSV file with LRU caching that invalidates when file mtime changes.

    This keeps the existing simple API (single `file_path` argument) but
    ensures cached values reflect the current file contents when the file
    is modified on disk.
    """
    path = Path(file_path)
    if not path.exists():
        return None
    try:
        # Use os.stat to get mtime; wrap in try/except in case of network filesystems
        mtime = None
        try:
            mtime = path.stat().st_mtime
        except Exception:
            # Fall back to os.path.getmtime if Path.stat() fails for any reason
            try:
                mtime = os.path.getmtime(str(path))
            except Exception:
                mtime = None
        return _read_csv_with_mtime(str(path), mtime)
    except Exception as e:
        LOGGER.error(f"Error preparing to read {path}: {e}")
        return None


# Expose cache methods from the underlying cached function for testing/monitoring
read_csv_cached.cache_clear = _read_csv_with_mtime.cache_clear
read_csv_cached.cache_info = _read_csv_with_mtime.cache_info


def build_visibility_path(base_dir: Path, star_name: str, target_name: str) -> Path:
    """Build consistent visibility file path for planets.
    
    Pattern: base_dir / star_name / target_name / "Visibility for {target_name}.csv"
    Used throughout scheduler to avoid repeated f-string formatting.
    
    Args:
        base_dir: Base directory (e.g., output/data/targets)
        star_name: Name of the star
        target_name: Name of the planet/target
        
    Returns:
        Path to visibility file
    """
    return base_dir / star_name / target_name / f"Visibility for {target_name}.csv"


def build_star_visibility_path(base_dir: Path, star_name: str) -> Path:
    """Build visibility path for a star (no planet subdirectory).
    
    Pattern: base_dir / star_name / "Visibility for {star_name}.csv"
    
    Args:
        base_dir: Base directory (e.g., output/data/aux_targets)
        star_name: Name of the star
        
    Returns:
        Path to visibility file
    """
    return base_dir / star_name / f"Visibility for {star_name}.csv"


def read_star_visibility_cached(base_dir: Path, star_name: str) -> Optional[pd.DataFrame]:
    """Read star visibility file with caching.
    
    Convenience wrapper that combines path building and cached reading.
    
    Args:
        base_dir: Base directory (e.g., output/data/aux_targets)
        star_name: Name of the star
        
    Returns:
        DataFrame with visibility data or None if file doesn't exist
    """
    path = build_star_visibility_path(base_dir, star_name)
    return read_csv_cached(str(path))


def read_planet_visibility_cached(
    base_dir: Path, 
    star_name: str, 
    planet_name: str
) -> Optional[pd.DataFrame]:
    """Read planet visibility file with caching.
    
    Convenience wrapper that combines path building and cached reading.
    
    Args:
        base_dir: Base directory (e.g., output/data/targets)
        star_name: Name of the star
        planet_name: Name of the planet
        
    Returns:
        DataFrame with transit visibility data or None if file doesn't exist
    """
    path = build_visibility_path(base_dir, star_name, planet_name)
    return read_csv_cached(str(path))
