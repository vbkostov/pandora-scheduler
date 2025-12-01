"""Unified configuration system for Pandora Scheduler.

This module consolidates the scattered configuration classes into a single,
hierarchical system that's easier to understand and maintain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PandoraSchedulerConfig:
    """Master configuration for the Pandora Scheduler pipeline.
    
    This consolidates SchedulerConfig, ScienceCalendarConfig, and VisibilityConfig
    into a single, coherent configuration object.
    """
    
    # ============================================================================
    # TIMING & WINDOWS
    # ============================================================================
    
    window_start: datetime
    """Start of the scheduling window."""
    
    window_end: datetime
    """End of the scheduling window."""
    
    obs_window: timedelta = timedelta(hours=24)
    """Observation window size for scheduling (default: 24 hours)."""
    
    commissioning_days: int = 0
    """Number of commissioning days at start of mission."""
    
    # ============================================================================
    # PATHS & DATA SOURCES
    # ============================================================================
    
    targets_manifest: Optional[Path] = None
    """Path to target definition manifest/directory."""
    
    gmat_ephemeris: Optional[Path] = None
    """Path to GMAT ephemeris file (for visibility generation)."""
    
    output_dir: Optional[Path] = None
    """Output directory for generated files."""
    
    # ============================================================================
    # SCHEDULING THRESHOLDS
    # ============================================================================
    
    transit_coverage_min: float = 0.2
    """Minimum transit coverage to schedule (0-1). Lower = more transits scheduled."""
    
    min_visibility: float = 0.0
    """Minimum visibility fraction to consider observable."""
    
    deprioritization_limit_hours: float = 48.0
    """Deprioritize auxiliary targets after this many hours of observation."""
    
    saa_overlap_threshold: float = 0.0
    """Maximum acceptable SAA overlap fraction (0-1)."""
    
    # ============================================================================
    # WEIGHTING FACTORS (must sum to 1.0)
    # ============================================================================
    
    sched_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)
    """Scheduling weights: (transit_coverage, saa_overlap, schedule_factor)."""
    
    calendar_weights: Tuple[float, float, float] = (0.8, 0.0, 0.2)
    """Calendar generation weights: (coverage, saa, schedule)."""
    
    # ============================================================================
    # KEEPOUT ANGLES (degrees)
    # ============================================================================
    
    sun_avoidance_deg: float = 91.0
    """Minimum angle from Sun (degrees)."""
    
    moon_avoidance_deg: float = 25.0
    """Minimum angle from Moon (degrees)."""
    
    earth_avoidance_deg: float = 86.0
    """Minimum angle from Earth limb (degrees)."""
    
    # ============================================================================
    # XML GENERATION PARAMETERS
    # ============================================================================
    
    obs_sequence_duration_min: int = 90
    """Observation sequence duration in minutes."""
    
    occ_sequence_limit_min: int = 50
    """Maximum occultation sequence duration in minutes."""
    
    min_sequence_minutes: int = 5
    """Minimum sequence length to include in XML (shorter sequences dropped)."""
    
    break_occultation_sequences: bool = True
    """Break long occultation sequences into chunks."""
    
    # ============================================================================
    # STANDARD OBSERVATIONS
    # ============================================================================
    
    std_obs_duration_hours: float = 0.5
    """Duration of standard star observations in hours."""
    
    std_obs_frequency_days: float = 3.0
    """Frequency of standard star observations in days."""
    
    # ============================================================================
    # BEHAVIOR FLAGS
    # ============================================================================
    
    show_progress: bool = False
    """Show progress bars during processing."""
    
    force_regenerate: bool = False
    """Force regeneration of files even if they exist."""
    
    use_target_list_for_occultations: bool = False
    """Use target list for occultation scheduling (vs. separate list)."""
    
    prioritise_occultations_by_slew: bool = False
    """Prioritize occultation targets by slew angle."""
    
    # ============================================================================
    # AUXILIARY SORTING
    # ============================================================================
    
    aux_sort_key: str = "sort_by_tdf_priority"
    """Key for sorting auxiliary targets."""
    
    # ============================================================================
    # METADATA
    # ============================================================================
    
    author: Optional[str] = None
    """Author name for XML metadata."""
    
    created_timestamp: Optional[datetime | str] = None
    """Creation timestamp for XML metadata."""
    
    visit_limit: Optional[int] = None
    """Limit number of visits (for testing). None = no limit."""
    
    target_filters: Sequence[str] = field(default_factory=tuple)
    """Target name filters for visibility generation."""
    
    extra_inputs: Dict[str, Path] = field(default_factory=dict)
    """Additional input files (auxiliary lists, etc.)."""
    
    # ============================================================================
    # VALIDATION
    # ============================================================================
    
    def __post_init__(self) -> None:
        """Validate configuration consistency."""
        # Validate sched_weights sum to 1.0
        if not np.isclose(sum(self.sched_weights), 1.0):
            raise ValueError(
                f"sched_weights must sum to 1.0, got {sum(self.sched_weights)}"
            )
        
        # Validate calendar_weights sum to 1.0
        if not np.isclose(sum(self.calendar_weights), 1.0):
            raise ValueError(
                f"calendar_weights must sum to 1.0, got {sum(self.calendar_weights)}"
            )
        
        # Validate transit_coverage_min in range
        if not 0.0 <= self.transit_coverage_min <= 1.0:
            raise ValueError(
                f"transit_coverage_min must be in [0, 1], got {self.transit_coverage_min}"
            )
    
    # ============================================================================
    # CONVERSION METHODS (for backward compatibility)
    # ============================================================================
    
    def to_scheduler_config(self):
        """Convert to legacy SchedulerConfig format."""
        from pandorascheduler_rework.scheduler import SchedulerConfig
        
        return SchedulerConfig(
            obs_window=self.obs_window,
            transit_coverage_min=self.transit_coverage_min,
            sched_weights=self.sched_weights,
            min_visibility=self.min_visibility,
            deprioritization_limit_hours=self.deprioritization_limit_hours,
            commissioning_days=self.commissioning_days,
            aux_key=self.aux_sort_key,
            show_progress=self.show_progress,
            std_obs_duration_hours=self.std_obs_duration_hours,
            std_obs_frequency_days=self.std_obs_frequency_days,
        )
    
    def to_science_calendar_config(self):
        """Convert to legacy ScienceCalendarConfig format."""
        from pandorascheduler_rework.science_calendar import ScienceCalendarConfig
        
        return ScienceCalendarConfig(
            visit_limit=self.visit_limit,
            obs_sequence_duration_min=self.obs_sequence_duration_min,
            occ_sequence_limit_min=self.occ_sequence_limit_min,
            min_sequence_minutes=self.min_sequence_minutes,
            break_occultation_sequences=self.break_occultation_sequences,
            use_target_list_for_occultations=self.use_target_list_for_occultations,
            prioritise_occultations_by_slew=self.prioritise_occultations_by_slew,
            calendar_weights=self.calendar_weights,
            keepout_angles=(
                self.sun_avoidance_deg,
                self.moon_avoidance_deg,
                self.earth_avoidance_deg,
            ),
            author=self.author,
            show_progress=self.show_progress,
            created_timestamp=self.created_timestamp,
        )
    
    def to_visibility_config(self):
        """Convert to legacy VisibilityConfig format."""
        from pandorascheduler_rework.visibility.config import VisibilityConfig
        
        if self.targets_manifest is None:
            raise ValueError("targets_manifest required for VisibilityConfig")
        if self.gmat_ephemeris is None:
            raise ValueError("gmat_ephemeris required for VisibilityConfig")
        
        return VisibilityConfig(
            window_start=self.window_start,
            window_end=self.window_end,
            gmat_ephemeris=self.gmat_ephemeris,
            target_list=self.targets_manifest,
            partner_list=None,  # Can be added to PandoraSchedulerConfig if needed
            output_root=self.output_dir,
            sun_avoidance_deg=self.sun_avoidance_deg,
            moon_avoidance_deg=self.moon_avoidance_deg,
            earth_avoidance_deg=self.earth_avoidance_deg,
            force=self.force_regenerate,
            target_filters=self.target_filters,
        )
