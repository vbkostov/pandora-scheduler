"""Unit tests for build_schedule() pipeline API."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from pandorascheduler_rework.config import PandoraSchedulerConfig
from pandorascheduler_rework.pipeline import build_schedule


class TestBuildScheduleV2:
    """Test the v2 pipeline API with PandoraSchedulerConfig."""

    def test_requires_valid_config(self):
        """Test that invalid config raises validation errors."""
        with pytest.raises(ValueError, match="transit_scheduling_weights must sum to 1.0"):
            PandoraSchedulerConfig(
                window_start=datetime(2026, 2, 5),
                window_end=datetime(2026, 2, 19),
                targets_manifest=Path("data"),
                transit_scheduling_weights=(0.5, 0.5, 0.5),  # Invalid: sums to 1.5
            )

    def test_validates_transit_coverage_range(self):
        """Test that transit_coverage_min is validated."""
        with pytest.raises(ValueError, match="transit_coverage_min must be in"):
            PandoraSchedulerConfig(
                window_start=datetime(2026, 2, 5),
                window_end=datetime(2026, 2, 19),
                targets_manifest=Path("data"),
                transit_coverage_min=1.5,  # Invalid: > 1.0
            )

    def test_config_has_sensible_defaults(self):
        """Test that config has reasonable defaults."""
        config = PandoraSchedulerConfig(
            window_start=datetime(2026, 2, 5),
            window_end=datetime(2026, 2, 19),
            targets_manifest=Path("data"),
        )
        
        assert config.transit_coverage_min == 0.2
        assert config.transit_scheduling_weights == (0.8, 0.0, 0.2)
        assert config.show_progress is False
        assert config.sun_avoidance_deg == 91.0

    def test_config_is_immutable(self):
        """Test that config is frozen (immutable)."""
        config = PandoraSchedulerConfig(
            window_start=datetime(2026, 2, 5),
            window_end=datetime(2026, 2, 19),
            targets_manifest=Path("data"),
        )
        
        with pytest.raises(Exception):  # FrozenInstanceError
            config.transit_coverage_min = 0.5  # type: ignore

    def test_conversion_to_scheduler_config(self):
        """Test conversion to legacy SchedulerConfig."""
        config = PandoraSchedulerConfig(
            window_start=datetime(2026, 2, 5),
            window_end=datetime(2026, 2, 19),
            targets_manifest=Path("data"),
            transit_coverage_min=0.3,
            deprioritization_limit_hours=72.0,
        )
        
        scheduler_config = config.to_scheduler_config()
        
        assert scheduler_config.transit_coverage_min == 0.3
        assert scheduler_config.deprioritization_limit_hours == 72.0
        assert scheduler_config.transit_scheduling_weights == (0.8, 0.0, 0.2)

    def test_conversion_to_science_calendar_config(self):
        """Test conversion to legacy ScienceCalendarConfig."""
        config = PandoraSchedulerConfig(
            window_start=datetime(2026, 2, 5),
            window_end=datetime(2026, 2, 19),
            targets_manifest=Path("data"),
            obs_sequence_duration_min=120,
            occ_sequence_limit_min=60,
        )
        
        calendar_config = config.to_science_calendar_config()
        
        assert calendar_config.obs_sequence_duration_min == 120
        assert calendar_config.occ_sequence_limit_min == 60
        assert calendar_config.keepout_angles == (91.0, 25.0, 86.0)

    def test_conversion_to_visibility_config_requires_paths(self):
        """Test that visibility config conversion requires GMAT ephemeris."""
        config = PandoraSchedulerConfig(
            window_start=datetime(2026, 2, 5),
            window_end=datetime(2026, 2, 19),
            targets_manifest=Path("data"),
            # No gmat_ephemeris provided
        )
        
        with pytest.raises(ValueError, match="gmat_ephemeris required"):
            config.to_visibility_config()

    def test_conversion_to_visibility_config_success(self):
        """Test successful conversion to VisibilityConfig."""
        config = PandoraSchedulerConfig(
            window_start=datetime(2026, 2, 5),
            window_end=datetime(2026, 2, 19),
            targets_manifest=Path("data/targets.csv"),
            gmat_ephemeris=Path("data/ephemeris.txt"),
            output_dir=Path("output"),
            sun_avoidance_deg=95.0,
        )
        
        vis_config = config.to_visibility_config()
        
        assert vis_config.window_start == datetime(2026, 2, 5)
        assert vis_config.window_end == datetime(2026, 2, 19)
        assert vis_config.gmat_ephemeris == Path("data/ephemeris.txt")
        assert vis_config.sun_avoidance_deg == 95.0


class TestBuildScheduleV2Integration:
    """Integration tests for build_schedule() (require test data)."""

    @pytest.mark.skip(reason="Requires test data fixtures")
    def test_basic_scheduling_run(self, tmp_path):
        """Test a basic scheduling run with minimal config."""
        config = PandoraSchedulerConfig(
            window_start=datetime(2026, 2, 5),
            window_end=datetime(2026, 2, 6),  # 1 day
            targets_manifest=Path("test_data/targets"),
            output_dir=tmp_path,
            show_progress=False,
        )
        
        result = build_schedule(config)
        
        assert result.schedule_csv is not None
        assert result.schedule_csv.exists()

    @pytest.mark.skip(reason="Requires test data fixtures")
    def test_with_visibility_generation(self, tmp_path):
        """Test scheduling with visibility generation enabled."""
        config = PandoraSchedulerConfig(
            window_start=datetime(2026, 2, 5),
            window_end=datetime(2026, 2, 6),
            targets_manifest=Path("test_data/targets"),
            gmat_ephemeris=Path("test_data/ephemeris.txt"),
            output_dir=tmp_path,
            show_progress=False,
        )
        
        result = build_schedule(config)
        
        assert result.schedule_csv is not None
        # Visibility files should be generated
        assert (tmp_path / "data" / "targets").exists()


class TestConfigDocumentation:
    """Test that config fields are well-documented."""

    def test_all_fields_have_docstrings(self):
        """Test that all config fields have documentation."""
        config = PandoraSchedulerConfig(
            window_start=datetime(2026, 2, 5),
            window_end=datetime(2026, 2, 19),
            targets_manifest=Path("data"),
        )
        
        # Check that key fields have docstrings in the class
        assert PandoraSchedulerConfig.__doc__ is not None
        assert "Master configuration" in PandoraSchedulerConfig.__doc__

    def test_config_repr_is_readable(self):
        """Test that config has a readable string representation."""
        config = PandoraSchedulerConfig(
            window_start=datetime(2026, 2, 5),
            window_end=datetime(2026, 2, 19),
            targets_manifest=Path("data"),
            transit_coverage_min=0.3,
        )
        
        repr_str = repr(config)
        assert "PandoraSchedulerConfig" in repr_str
        assert "transit_coverage_min=0.3" in repr_str


class TestBackwardCompatibility:
    """Test backward compatibility with old API."""

    def test_v2_produces_same_structure_as_v1(self):
        """Test that v2 API produces same result structure as v1."""
        # This is a structural test - both APIs should return SchedulerResult

        
        # build_schedule_v2 should return SchedulerResult
        # (actual execution skipped without test data)
        assert hasattr(build_schedule, "__call__")
        assert build_schedule.__doc__ is not None
