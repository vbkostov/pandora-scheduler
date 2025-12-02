"""High-level entry points for the reworked scheduler pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from pandorascheduler_rework import observation_utils as rework_helper
from pandorascheduler_rework.utils.io import read_csv_cached
from pandorascheduler_rework.config import PandoraSchedulerConfig
from pandorascheduler_rework.scheduler import (
    SchedulerConfig,
    SchedulerInputs,
    SchedulerPaths,
    run_scheduler,
)
from pandorascheduler_rework.visibility.catalog import build_visibility_catalog
from pandorascheduler_rework.visibility.config import VisibilityConfig


LOGGER = logging.getLogger(__name__)


# SchedulerRequest removed (legacy)


@dataclass
class SchedulerResult:
    """Outputs produced by a scheduling run.

    Maintaining a structured return value simplifies comparisons with the
    legacy scheduler while leaving room to attach additional diagnostics later
    (plots, metrics, logs, ...).
    """

    schedule_csv: Optional[Path] = None
    reports: Dict[str, Path] = field(default_factory=dict)
    diagnostics: Dict[str, object] = field(default_factory=dict)

    def iter_output_files(self) -> Iterable[Path]:
        """Yield every concrete file generated during the run."""

        if self.schedule_csv:
            yield self.schedule_csv
        for path in self.reports.values():
            yield path


def build_schedule(config: PandoraSchedulerConfig) -> SchedulerResult:
    """Run the modern scheduler and persist outputs alongside diagnostics.
    
    Args:
        config: Unified configuration object
        
    Returns:
        SchedulerResult with paths to generated files
    """
    if config.output_dir is None:
        # Should be caught by config validation or caller, but safe fallback
        output_dir = Path("output")
    else:
        output_dir = config.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Filesystem layout for this run. Use the run's `output_dir` as the
    # package root so generated data lives under `<output_dir>/data` which
    # matches the expectations of downstream components.
    paths = SchedulerPaths.from_package_root(output_dir)


    # Prepare extra_inputs
    extra_inputs = config.extra_inputs

    # When generating target manifests from a provided target definition base,
    # write the generated CSV manifests into the run's output data directory so
    # subsequent steps (visibility & calendar generation) can find them there.
    out_data = output_dir / "data"
    
    # We need to resolve these paths now to pass to target manifest generation
    # IMPORTANT: Use absolute paths to avoid legacy folder lookups
    primary_target_csv = _coerce_path(
        extra_inputs.get("primary_target_csv"),
        out_data / "exoplanet_targets.csv",
    ).resolve()
    auxiliary_target_csv = _coerce_path(
        extra_inputs.get("auxiliary_target_csv"),
        out_data / "auxiliary-standard_targets.csv",
    ).resolve()
    monitoring_target_csv = _coerce_path(
        extra_inputs.get("monitoring_target_csv"),
        out_data / "monitoring-standard_targets.csv",
    ).resolve()
    occultation_target_csv = _coerce_path(
        extra_inputs.get("occultation_target_csv"),
        out_data / "occultation-standard_targets.csv",
    ).resolve()

    # Handle target definition files if provided
    target_definition_files_raw = extra_inputs.get("target_definition_files")
    if target_definition_files_raw:
        target_definition_files = _resolve_target_definition_files(
            target_definition_files_raw,
            [
                _target_definition_from_csv(primary_target_csv),
                _target_definition_from_csv(auxiliary_target_csv),
                _target_definition_from_csv(monitoring_target_csv),
                _target_definition_from_csv(occultation_target_csv),
            ],
        )
        
        target_definition_base = _coerce_optional_path(
            extra_inputs.get("target_definition_base")
        )

        if target_definition_base is not None:
            # Allow callers to opt-out of regenerating manifests (use existing CSVs)
            skip_manifests = _as_bool(extra_inputs.get("skip_manifests"), False)
            if skip_manifests:
                LOGGER.info("Skipping generation of target manifests (skip_manifests=True)")
            else:
                _generate_target_manifests(
                    target_definition_files,
                    target_definition_base,
                    primary_target_csv,
                    auxiliary_target_csv,
                    monitoring_target_csv,
                    occultation_target_csv,
                )
    else:
        # Fallback if not explicitly provided (legacy behavior relied on implicit paths)
        # But for V2 we prefer explicit. If missing, we assume the CSVs exist.
        target_definition_files = [
            _target_definition_from_csv(primary_target_csv),
            _target_definition_from_csv(auxiliary_target_csv),
            _target_definition_from_csv(monitoring_target_csv),
            _target_definition_from_csv(occultation_target_csv),
        ]

    if config.targets_manifest and not config.targets_manifest.exists():
        raise FileNotFoundError(f"Provided targets_manifest not found: {config.targets_manifest}")

    _maybe_generate_visibility(
        config,
        paths,
        config.window_start,
        config.window_end,
        primary_target_csv,
        auxiliary_target_csv,
        monitoring_target_csv,
        occultation_target_csv,
    )

    target_list = read_csv_cached(str(primary_target_csv))
    if target_list is None:
        raise FileNotFoundError(f"Primary target manifest not found: {primary_target_csv}")

    scheduler_inputs = SchedulerInputs(
        pandora_start=config.window_start,
        pandora_stop=config.window_end,
        sched_start=config.window_start,
        sched_stop=config.window_end,
        target_list=target_list,
        paths=paths,
        target_definition_files=target_definition_files,
        primary_target_csv=primary_target_csv,
        auxiliary_target_csv=auxiliary_target_csv,
        occultation_target_csv=occultation_target_csv,
        output_dir=output_dir,
        tracker_pickle_path=_coerce_optional_path(extra_inputs.get("tracker_pickle")),
    )

    scheduler_config = config.to_scheduler_config()

    outputs = run_scheduler(scheduler_inputs, scheduler_config)

    reports: Dict[str, Path] = {}
    if outputs.observation_report_path is not None:
        reports["observation_time"] = outputs.observation_report_path
    if outputs.tracker_csv_path is not None:
        reports["tracker_csv"] = outputs.tracker_csv_path
    if outputs.tracker_pickle_path is not None:
        reports["tracker_pickle"] = outputs.tracker_pickle_path

    diagnostics: Dict[str, Any] = {
        "schedule_dataframe": outputs.schedule,
        "tracker_dataframe": outputs.tracker,
    }

    return SchedulerResult(
        schedule_csv=outputs.schedule_path,
        reports=reports,
        diagnostics=diagnostics,
    )


def _maybe_generate_visibility(
    config: PandoraSchedulerConfig,
    paths: SchedulerPaths,
    pandora_start: datetime,
    pandora_stop: datetime,
    primary_target_csv: Path,
    auxiliary_target_csv: Path,
    monitoring_target_csv: Path,
    occultation_target_csv: Path,
) -> None:
    # Determine whether we should generate visibility. Default: True when a GMAT ephemeris
    # is provided, or when explicitly requested via extra_inputs.
    generate_visibility = bool(config.gmat_ephemeris) or bool(
        str(config.extra_inputs.get("generate_visibility", "")).lower() in {"1", "true", "yes", "y"}
    )

    if not generate_visibility:
        return

    # 1. Primary Targets -> data/targets
    primary_config = _build_visibility_config(
        config,
        paths,
        pandora_start,
        pandora_stop,
        target_list=primary_target_csv,
        partner_list=auxiliary_target_csv,
        output_subpath="targets",
    )
    LOGGER.info("Generating visibility for Primary Targets in %s", primary_config.output_root)
    build_visibility_catalog(primary_config)

    # 2. Auxiliary Targets -> data/aux_targets
    aux_config = _build_visibility_config(
        config,
        paths,
        pandora_start,
        pandora_stop,
        target_list=auxiliary_target_csv,
        partner_list=None,
        output_subpath="aux_targets",
    )
    LOGGER.info("Generating visibility for Auxiliary Targets in %s", aux_config.output_root)
    build_visibility_catalog(aux_config)

    # 3. Monitoring Targets -> data/aux_targets
    mon_config = _build_visibility_config(
        config,
        paths,
        pandora_start,
        pandora_stop,
        target_list=monitoring_target_csv,
        partner_list=None,
        output_subpath="aux_targets",
    )
    LOGGER.info("Generating visibility for Monitoring Targets in %s", mon_config.output_root)
    build_visibility_catalog(mon_config)

    # 3. Occultation Targets -> data/aux_targets
    occ_config = _build_visibility_config(
        config,
        paths,
        pandora_start,
        pandora_stop,
        target_list=occultation_target_csv,
        partner_list=None,
        output_subpath="aux_targets",
    )
    LOGGER.info("Generating visibility for Occultation Targets in %s", occ_config.output_root)
    build_visibility_catalog(occ_config)


def _generate_target_manifests(
    target_definition_files: Sequence[str],
    base_dir: Path,
    primary_target_csv: Path,
    auxiliary_target_csv: Path,
    monitoring_target_csv: Path,
    occultation_target_csv: Path,
) -> None:
    mapping = {
        0: primary_target_csv,
        1: auxiliary_target_csv,
        2: monitoring_target_csv,
        3: occultation_target_csv,
    }

    for index, category in enumerate(target_definition_files):
        destination = mapping.get(index)
        if destination is None:
            continue

        manifest = rework_helper.process_target_files(
            category,
            base_path=base_dir,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(destination, index=False)

    rework_helper.create_aux_list(
        target_definition_files,
        primary_target_csv.parent.parent,
    )

def _build_visibility_config(
    config: PandoraSchedulerConfig,
    paths: SchedulerPaths,
    pandora_start: datetime,
    pandora_stop: datetime,
    target_list: Path,
    partner_list: Optional[Path],
    output_subpath: str,
) -> VisibilityConfig:
    extra_inputs = config.extra_inputs

    gmat_path = config.gmat_ephemeris or extra_inputs.get("visibility_gmat")
    if gmat_path is None:
        raise ValueError("gmat_ephemeris required for visibility generation")

    target_list_path = extra_inputs.get("visibility_target_list") or target_list

    partner_override = extra_inputs.get("visibility_partner_list")
    
    if partner_override is not None:
        partner_list = partner_override
    else:
        # Use the passed partner_list argument as-is (which is usually auxiliary_target_csv)
        pass

    output_override = extra_inputs.get("visibility_output_root")
    if output_override is not None:
        output_root = output_override
    else:
        # Use config output dir + subpath
        if config.output_dir:
            output_root = config.output_dir / "data" / output_subpath
        else:
            output_root = Path("output") / "data" / output_subpath

    return VisibilityConfig(
        window_start=pandora_start,
        window_end=pandora_stop,
        gmat_ephemeris=gmat_path,
        target_list=target_list_path,
        partner_list=partner_list,
        output_root=output_root,
        sun_avoidance_deg=config.sun_avoidance_deg,
        moon_avoidance_deg=config.moon_avoidance_deg,
        earth_avoidance_deg=config.earth_avoidance_deg,
        force=config.force_regenerate,
        target_filters=config.target_filters,
    )


# _build_config removed (legacy)


def _coerce_datetime(value: object, default: datetime) -> datetime:
    if value is None:
        return default
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError(f"Unsupported datetime value: {value!r}")


def _coerce_path(value: object, default: Path) -> Path:
    if value is None:
        return default
    return Path(str(value)).expanduser().resolve()


def _coerce_optional_path(value: object) -> Optional[Path]:
    if value is None:
        return None
    return Path(str(value)).expanduser().resolve()


def _resolve_target_definition_files(
    value: object, fallback: Sequence[str]
) -> List[str]:
    if value is None:
        return list(fallback)
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    raise TypeError(
        "target_definition_files must be a sequence or comma-separated string"
    )


def _target_definition_from_csv(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_targets"):
        stem = stem[: -len("_targets")]
    return stem


def _coerce_transit_scheduling_weights(value: object) -> tuple[float, float, float]:
    components: List[object]
    if isinstance(value, str):
        components = [
            component.strip() for component in value.split(",") if component.strip()
        ]
    elif isinstance(value, (list, tuple)):
        components = list(value)
    elif value is None:
        components = []
    else:
        components = [value]

    if not components:
        raise ValueError("transit_scheduling_weights must provide three numeric components")

    weights = tuple(_as_float(component, 0.0) for component in components)

    if len(weights) != 3:
        raise ValueError(
            "transit_scheduling_weights must contain exactly three values (coverage, saa, schedule)."
        )
    return (weights[0], weights[1], weights[2])


def _as_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "":
            return default
        return normalized in {"1", "true", "yes", "y", "on"}
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _coerce_target_filters(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        tokens = [item.strip() for item in value.split(",") if item.strip()]
        return tuple(tokens)
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value)
    return (str(value),)


def _as_float(value: object, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected float-like value, received {value!r}")


def _as_int(value: object, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"Expected int-like value, received {value!r}")
