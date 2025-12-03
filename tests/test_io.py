"""Unit tests for utils.io module."""

from pathlib import Path

import pandas as pd

from pandorascheduler_rework.utils.io import (
    build_star_visibility_path,
    build_visibility_path,
    read_csv_cached,
    read_planet_visibility_cached,
    read_star_visibility_cached,
)


class TestPathBuilding:
    """Test visibility path construction functions."""

    def test_build_visibility_path_planet(self):
        """Test planet visibility path construction."""
        base = Path("/data/targets")
        star = "TOI-700"
        planet = "TOI-700 b"
        
        result = build_visibility_path(base, star, planet)
        
        assert result == Path("/data/targets/TOI-700/TOI-700 b/Visibility for TOI-700 b.csv")

    def test_build_visibility_path_with_spaces(self):
        """Test path construction with spaces in names."""
        base = Path("/output/data/targets")
        star = "HD 189733"
        planet = "HD 189733 b"
        
        result = build_visibility_path(base, star, planet)
        
        assert result == Path("/output/data/targets/HD 189733/HD 189733 b/Visibility for HD 189733 b.csv")

    def test_build_star_visibility_path(self):
        """Test star-only visibility path construction."""
        base = Path("/data/aux_targets")
        star = "Vega"
        
        result = build_star_visibility_path(base, star)
        
        assert result == Path("/data/aux_targets/Vega/Visibility for Vega.csv")

    def test_build_star_visibility_path_with_underscores(self):
        """Test star path with underscores."""
        base = Path("/output/data/aux_targets")
        star = "GJ_1214"
        
        result = build_star_visibility_path(base, star)
        
        assert result == Path("/output/data/aux_targets/GJ_1214/Visibility for GJ_1214.csv")


class TestReadCsvCached:
    """Test CSV reading with caching."""

    def test_read_csv_cached_success(self, tmp_path):
        """Test successful CSV reading."""
        # Create a test CSV
        csv_path = tmp_path / "test.csv"
        test_data = pd.DataFrame({
            "Time(MJD_UTC)": [59000.0, 59000.5, 59001.0],
            "Visible": [1, 0, 1],
        })
        test_data.to_csv(csv_path, index=False)
        
        # Read it
        result = read_csv_cached(str(csv_path))
        
        assert result is not None
        assert len(result) == 3
        assert "Time(MJD_UTC)" in result.columns
        assert "Visible" in result.columns

    def test_read_csv_cached_nonexistent_file(self):
        """Test reading non-existent file returns None."""
        result = read_csv_cached("/nonexistent/path/file.csv")
        
        assert result is None

    def test_read_csv_cached_caching_works(self, tmp_path):
        """Test that caching actually caches (same object returned)."""
        # Create a test CSV
        csv_path = tmp_path / "cache_test.csv"
        test_data = pd.DataFrame({"A": [1, 2, 3]})
        test_data.to_csv(csv_path, index=False)
        
        # Read twice
        result1 = read_csv_cached(str(csv_path))
        result2 = read_csv_cached(str(csv_path))
        
        # Should be the exact same object (cached)
        assert result1 is result2

    def test_read_csv_cached_different_paths_different_cache(self, tmp_path):
        """Test that different paths get different cache entries."""
        # Create two test CSVs
        csv1 = tmp_path / "file1.csv"
        csv2 = tmp_path / "file2.csv"
        
        pd.DataFrame({"A": [1]}).to_csv(csv1, index=False)
        pd.DataFrame({"B": [2]}).to_csv(csv2, index=False)
        
        result1 = read_csv_cached(str(csv1))
        result2 = read_csv_cached(str(csv2))
        
        assert result1 is not result2
        assert "A" in result1.columns
        assert "B" in result2.columns


class TestConvenienceWrappers:
    """Test convenience wrapper functions."""

    def test_read_star_visibility_cached(self, tmp_path):
        """Test star visibility reading wrapper."""
        # Create directory structure
        star_dir = tmp_path / "Vega"
        star_dir.mkdir()
        
        vis_file = star_dir / "Visibility for Vega.csv"
        test_data = pd.DataFrame({
            "Time(MJD_UTC)": [59000.0],
            "Visible": [1],
        })
        test_data.to_csv(vis_file, index=False)
        
        # Read using wrapper
        result = read_star_visibility_cached(tmp_path, "Vega")
        
        assert result is not None
        assert len(result) == 1
        assert result["Visible"].iloc[0] == 1

    def test_read_star_visibility_cached_missing_file(self, tmp_path):
        """Test star visibility wrapper with missing file."""
        result = read_star_visibility_cached(tmp_path, "NonexistentStar")
        
        assert result is None

    def test_read_planet_visibility_cached(self, tmp_path):
        """Test planet visibility reading wrapper."""
        # Create directory structure
        star_dir = tmp_path / "TOI-700"
        planet_dir = star_dir / "TOI-700 b"
        planet_dir.mkdir(parents=True)
        
        vis_file = planet_dir / "Visibility for TOI-700 b.csv"
        test_data = pd.DataFrame({
            "Transit_Start": [59000.0],
            "Transit_Stop": [59000.1],
            "Transit_Coverage": [0.95],
        })
        test_data.to_csv(vis_file, index=False)
        
        # Read using wrapper
        result = read_planet_visibility_cached(tmp_path, "TOI-700", "TOI-700 b")
        
        assert result is not None
        assert len(result) == 1
        assert result["Transit_Coverage"].iloc[0] == 0.95

    def test_read_planet_visibility_cached_missing_file(self, tmp_path):
        """Test planet visibility wrapper with missing file."""
        result = read_planet_visibility_cached(tmp_path, "Star", "Planet")
        
        assert result is None


class TestCachePerformance:
    """Test caching behavior and performance characteristics."""

    def test_cache_info_accessible(self, tmp_path):
        """Test that cache info is accessible for monitoring."""
        # Clear cache first
        read_csv_cached.cache_clear()
        
        # Create and read a file
        csv_path = tmp_path / "perf_test.csv"
        pd.DataFrame({"A": [1]}).to_csv(csv_path, index=False)
        
        read_csv_cached(str(csv_path))
        
        # Check cache info
        info = read_csv_cached.cache_info()
        assert info.hits == 0  # First read
        assert info.misses == 1
        
        # Read again
        read_csv_cached(str(csv_path))
        info = read_csv_cached.cache_info()
        assert info.hits == 1  # Second read from cache
        assert info.misses == 1

    def test_cache_can_be_cleared(self, tmp_path):
        """Test that cache can be manually cleared."""
        csv_path = tmp_path / "clear_test.csv"
        pd.DataFrame({"A": [1]}).to_csv(csv_path, index=False)
        
        # Read and cache
        result1 = read_csv_cached(str(csv_path))
        
        # Clear cache
        read_csv_cached.cache_clear()
        
        # Read again - should be different object
        result2 = read_csv_cached(str(csv_path))
        
        # Values should be equal but not same object
        pd.testing.assert_frame_equal(result1, result2)
        assert result1 is not result2  # Different objects after cache clear
