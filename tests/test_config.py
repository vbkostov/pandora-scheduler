"""Unit tests for unified configuration system."""

import pytest
from datetime import datetime, timedelta
from pathlib import Path

from pandorascheduler_rework.config import PandoraSchedulerConfig


class TestPandoraSchedulerConfig:
    """Tests for unified configuration class."""
    
    def test_minimal_config(self):
        """Test creating config with minimal required parameters."""
        config = PandoraSchedulerConfig(
            window_start=datetime(2026, 2, 5),
            window_end=datetime(2027, 2, 5),
        )
        
        assert config.window_start == datetime(2026, 2, 5)
        assert config.window_end == datetime(2027, 2, 5)
        assert config.obs_window == timedelta(hours=24)  # Default
        
    def test_full_config(self):
        """Test creating config with all parameters."""
        config = PandoraSchedulerConfig(
            window_start=datetime(2026, 2, 5),
            window_end=datetime(2027, 2, 5),
            targets_manifest=Path("targets"),
            gmat_ephemeris=Path("ephemeris.csv"),
            output_dir=Path("output"),
            transit_coverage_min=0.3,
            sched_weights=(0.6, 0.2, 0.2),
            show_progress=True,
        )
        
        assert config.transit_coverage_min == 0.3
        assert config.sched_weights == (0.6, 0.2, 0.2)
        assert config.show_progress is True
        
    def test_sched_weights_validation_pass(self):
        """Test that sched_weights summing to 1.0 validates."""
        config = PandoraSchedulerConfig(
            window_start=datetime(2026, 2, 5),
            window_end=datetime(2027, 2, 5),
            sched_weights=(0.5, 0.3, 0.2),  # Sums to 1.0
        )
        assert sum(config.sched_weights) == pytest.approx(1.0)
        
    def test_sched_weights_validation_fail(self):
        """Test that sched_weights not summing to 1.0 raises error."""
        with pytest.raises(ValueError, match="sched_weights must sum to 1.0"):
            PandoraSchedulerConfig(
                window_start=datetime(2026, 2, 5),
                window_end=datetime(2027, 2, 5),
                sched_weights=(0.5, 0.5, 0.5),  # Sums to 1.5!
            )
            
    def test_calendar_weights_validation_pass(self):
        """Test that calendar_weights summing to 1.0 validates."""
        config = PandoraSchedulerConfig(
            window_start=datetime(2026, 2, 5),
            window_end=datetime(2027, 2, 5),
            calendar_weights=(0.7, 0.1, 0.2),  # Sums to 1.0
        )
        assert sum(config.calendar_weights) == pytest.approx(1.0)
        
    def test_calendar_weights_validation_fail(self):
        """Test that calendar_weights not summing to 1.0 raises error."""
        with pytest.raises(ValueError, match="calendar_weights must sum to 1.0"):
            PandoraSchedulerConfig(
                window_start=datetime(2026, 2, 5),
                window_end=datetime(2027, 2, 5),
                calendar_weights=(1.0, 0.5, 0.5),  # Sums to 2.0!
            )
            
    def test_transit_coverage_min_validation_pass(self):
        """Test that transit_coverage_min in [0, 1] validates."""
        config = PandoraSchedulerConfig(
            window_start=datetime(2026, 2, 5),
            window_end=datetime(2027, 2, 5),
            transit_coverage_min=0.5,
        )
        assert config.transit_coverage_min == 0.5
        
    def test_transit_coverage_min_validation_fail_negative(self):
        """Test that negative transit_coverage_min raises error."""
        with pytest.raises(ValueError, match="transit_coverage_min must be in"):
            PandoraSchedulerConfig(
                window_start=datetime(2026, 2, 5),
                window_end=datetime(2027, 2, 5),
                transit_coverage_min=-0.1,
            )
            
    def test_transit_coverage_min_validation_fail_too_large(self):
        """Test that transit_coverage_min > 1.0 raises error."""
        with pytest.raises(ValueError, match="transit_coverage_min must be in"):
            PandoraSchedulerConfig(
                window_start=datetime(2026, 2, 5),
                window_end=datetime(2027, 2, 5),
                transit_coverage_min=1.5,
            )
            
    def test_to_scheduler_config(self):
        """Test conversion to SchedulerConfig."""
        config = PandoraSchedulerConfig(
            window_start=datetime(2026, 2, 5),
            window_end=datetime(2027, 2, 5),
            obs_window=timedelta(hours=48),
            transit_coverage_min=0.25,
            sched_weights=(0.6, 0.2, 0.2),
            min_visibility=0.1,
            deprioritization_limit_hours=36.0,
        )
        
        scheduler_config = config.to_scheduler_config()
        
        assert scheduler_config.obs_window == timedelta(hours=48)
        assert scheduler_config.transit_coverage_min == 0.25
        assert scheduler_config.sched_weights == (0.6, 0.2, 0.2)
        assert scheduler_config.min_visibility == 0.1
        assert scheduler_config.deprioritization_limit_hours == 36.0
        
    def test_to_science_calendar_config(self):
        """Test conversion to ScienceCalendarConfig."""
        config = PandoraSchedulerConfig(
            window_start=datetime(2026, 2, 5),
            window_end=datetime(2027, 2, 5),
            visit_limit=100,
            obs_sequence_duration_min=120,
            min_sequence_minutes=10,
            calendar_weights=(0.9, 0.0, 0.1),
            sun_avoidance_deg=95.0,
            moon_avoidance_deg=30.0,
            earth_avoidance_deg=90.0,
        )
        
        cal_config = config.to_science_calendar_config()
        
        assert cal_config.visit_limit == 100
        assert cal_config.obs_sequence_duration_min == 120
        assert cal_config.min_sequence_minutes == 10
        assert cal_config.calendar_weights == (0.9, 0.0, 0.1)
        assert cal_config.keepout_angles == (95.0, 30.0, 90.0)
        
    def test_to_visibility_config(self):
        """Test conversion to VisibilityConfig."""
        config = PandoraSchedulerConfig(
            window_start=datetime(2026, 2, 5),
            window_end=datetime(2027, 2, 5),
            targets_manifest=Path("targets.csv"),
            gmat_ephemeris=Path("ephemeris.csv"),
            output_dir=Path("output"),
            sun_avoidance_deg=92.0,
            moon_avoidance_deg=26.0,
            earth_avoidance_deg=87.0,
            force_regenerate=True,
        )
        
        vis_config = config.to_visibility_config()
        
        assert vis_config.window_start == datetime(2026, 2, 5)
        assert vis_config.window_end == datetime(2027, 2, 5)
        assert vis_config.gmat_ephemeris == Path("ephemeris.csv")
        assert vis_config.target_list == Path("targets.csv")
        assert vis_config.sun_avoidance_deg == 92.0
        assert vis_config.moon_avoidance_deg == 26.0
        assert vis_config.earth_avoidance_deg == 87.0
        assert vis_config.force is True
        
    def test_to_visibility_config_missing_manifest(self):
        """Test that to_visibility_config raises if targets_manifest is None."""
        config = PandoraSchedulerConfig(
            window_start=datetime(2026, 2, 5),
            window_end=datetime(2027, 2, 5),
            gmat_ephemeris=Path("ephemeris.csv"),
        )
        
        with pytest.raises(ValueError, match="targets_manifest required"):
            config.to_visibility_config()
            
    def test_to_visibility_config_missing_ephemeris(self):
        """Test that to_visibility_config raises if gmat_ephemeris is None."""
        config = PandoraSchedulerConfig(
            window_start=datetime(2026, 2, 5),
            window_end=datetime(2027, 2, 5),
            targets_manifest=Path("targets.csv"),
        )
        
        with pytest.raises(ValueError, match="gmat_ephemeris required"):
            config.to_visibility_config()
            
    def test_defaults_are_sensible(self):
        """Test that all default values are sensible."""
        config = PandoraSchedulerConfig(
            window_start=datetime(2026, 2, 5),
            window_end=datetime(2027, 2, 5),
        )
        
        # Defaults should be the same as in the classes they replace
        assert config.transit_coverage_min == 0.2
        assert config.sched_weights == (0.5, 0.3, 0.2)
        assert config.calendar_weights == (0.8, 0.0, 0.2)
        assert config.sun_avoidance_deg == 91.0
        assert config.moon_avoidance_deg == 25.0
        assert config.earth_avoidance_deg == 86.0
        assert config.obs_sequence_duration_min == 90
        assert config.show_progress is False
        
    def test_frozen_immutable(self):
        """Test that config is immutable (frozen dataclass)."""
        config = PandoraSchedulerConfig(
            window_start=datetime(2026, 2, 5),
            window_end=datetime(2027, 2, 5),
        )
        
        with pytest.raises(AttributeError):
            config.transit_coverage_min = 0.5  # Can't modify frozen dataclass
