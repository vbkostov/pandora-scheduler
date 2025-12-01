"""Input/Output utilities with caching."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=64)
def read_csv_cached(file_path: str) -> Optional[pd.DataFrame]:
    """Read CSV file with LRU caching (max ~1.5 GB memory).
    
    Args:
        file_path: String path to CSV file (must be string for caching)
    
    Returns:
        DataFrame or None if file doesn't exist or error occurs
    """
    path = Path(file_path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        LOGGER.error(f"Error reading {path}: {e}")
        return None


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
