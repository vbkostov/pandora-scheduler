"""Configuration dataclasses for the visibility reimplementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class VisibilityConfig:
    """User-facing knobs for generating visibility artifacts."""

    window_start: datetime
    window_end: datetime
    gmat_ephemeris: Path
    target_list: Path
    partner_list: Path | None = None
    output_root: Path | None = None
    sun_avoidance_deg: float = 91.0
    moon_avoidance_deg: float = 25.0
    earth_avoidance_deg: float = 86.0
    force: bool = False
    target_filters: Sequence[str] = field(default_factory=tuple)
    prefer_catalog_coordinates: bool = False

    def resolve_output_root(self, package_root: Path) -> Path:
        """Resolve the output directory, defaulting to the legacy layout."""

        if self.output_root is not None:
            return self.output_root
        return package_root / "data" / "targets"
