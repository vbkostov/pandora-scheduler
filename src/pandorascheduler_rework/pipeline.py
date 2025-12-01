"""High-level entry points for the reworked scheduler pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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


@dataclass(frozen=True)
class SchedulerRequest:
    """User-supplied inputs for a scheduling run.

    The rework aims to accept an explicit manifest of resources rather than
    relying on implicit directory layouts.  This dataclass keeps the contract
    small and predictable while allowing higher-level tooling to prepare the
    inputs however it wishes.
    """

    targets_manifest: Path
    """Path to a fingerprint or manifest describing target definitions."""

    window_start: datetime
    window_end: datetime
    output_dir: Path

    config: Dict[str, object] = field(default_factory=dict)
    """Optional tweakable knobs (readout modes, weighting factors, etc.)."""

    extra_inputs: Dict[str, Path] = field(default_factory=dict)
    """Any additional artifacts the scheduler requires (visibility tiles, TLM).

    The new implementation will ingest these explicitly so the pipeline stays
    hermetic and easy to test.
    """


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


def build_schedule(request: SchedulerRequest) -> SchedulerResult:
    """Run the modern scheduler and persist outputs alongside diagnostics."""



    request.output_dir.mkdir(parents=True, exist_ok=True)

    legacy_package_dir = Path(__file__).resolve().parents[1] / "pandorascheduler"
    paths = SchedulerPaths.from_package_root(legacy_package_dir)

    pandora_start = _coerce_datetime(
        request.config.get("pandora_start"),
        request.window_start,
    )
    pandora_stop = _coerce_datetime(
        request.config.get("pandora_stop"),
        request.window_end,
    )

    tracker_pickle_override = _coerce_optional_path(
        request.extra_inputs.get("tracker_pickle")
    )

    primary_target_csv = _coerce_path(
        request.extra_inputs.get("primary_target_csv"),
        paths.data_dir / "exoplanet_targets.csv",
    )
    auxiliary_target_csv = _coerce_path(
        request.extra_inputs.get("auxiliary_target_csv"),
        paths.data_dir / "auxiliary-standard_targets.csv",
    )
    monitoring_target_csv = _coerce_path(
        request.extra_inputs.get("monitoring_target_csv"),
        paths.data_dir / "monitoring-standard_targets.csv",
    )
    occultation_target_csv = _coerce_path(
        request.extra_inputs.get("occultation_target_csv"),
        paths.data_dir / "occultation-standard_targets.csv",
    )

    target_definition_files = _resolve_target_definition_files(
        request.extra_inputs.get("target_definition_files"),
        [
            _target_definition_from_csv(primary_target_csv),
            _target_definition_from_csv(auxiliary_target_csv),
            _target_definition_from_csv(monitoring_target_csv),
            _target_definition_from_csv(occultation_target_csv),
        ],
    )

    target_definition_base = _coerce_optional_path(
        request.extra_inputs.get("target_definition_base")
        or request.config.get("target_definition_base")
    )

    if target_definition_base is not None:
        _generate_target_manifests(
            target_definition_files,
            target_definition_base,
            primary_target_csv,
            auxiliary_target_csv,
            monitoring_target_csv,
            occultation_target_csv,
        )

    if not request.targets_manifest.exists():
        raise FileNotFoundError(
            f"Targets manifest does not exist: {request.targets_manifest}"
        )

    _maybe_generate_visibility(
        request,
        paths,
        pandora_start,
        pandora_stop,
        primary_target_csv,
        auxiliary_target_csv,
        monitoring_target_csv,
        occultation_target_csv,
    )

    target_list = read_csv_cached(str(primary_target_csv))
    if target_list is None:
        raise FileNotFoundError(f"Primary target manifest not found: {primary_target_csv}")

    scheduler_inputs = SchedulerInputs(
        pandora_start=pandora_start,
        pandora_stop=pandora_stop,
        sched_start=request.window_start,
        sched_stop=request.window_end,
        target_list=target_list,
        paths=paths,
        target_definition_files=target_definition_files,
        primary_target_csv=primary_target_csv,
        auxiliary_target_csv=auxiliary_target_csv,
        occultation_target_csv=occultation_target_csv,
        output_dir=request.output_dir,
        tracker_pickle_path=tracker_pickle_override,
    )

    scheduler_config = _build_config(request.config)

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


def build_schedule_v2(config: PandoraSchedulerConfig) -> SchedulerResult:
    """Run the scheduler with unified configuration (V2 API).
    
    This is the new, simplified API that uses PandoraSchedulerConfig instead of
    the dict-based SchedulerRequest. It provides clearer configuration with
    validation and better documentation.
    
    Args:
        config: Unified configuration object
        
    Returns:
        SchedulerResult with paths to generated files
        
    Example:
        >>> from pandorascheduler_rework.config import PandoraSchedulerConfig
        >>> from datetime import datetime
        >>> from pathlib import Path
        >>> 
        >>> config = PandoraSchedulerConfig(
        ...     window_start=datetime(2026, 2, 5),
        ...     window_end=datetime(2027, 2, 5),
        ...     targets_manifest=Path("../PandoraTargetList/target_definition_files"),
        ...     output_dir=Path("output"),
        ...     transit_coverage_min=0.2,
        ...     show_progress=True,
        ... )
        >>> result = build_schedule_v2(config)
    """
    # Convert unified config to old SchedulerRequest for now
    # This provides backward compatibility while we migrate

    # Determine whether we should generate visibility. Default: True when a GMAT ephemeris
    # is provided, or when explicitly requested via extra_inputs.
    generate_visibility = bool(config.gmat_ephemeris) or bool(
        str(config.extra_inputs.get("generate_visibility", "")).lower() in {"1", "true", "yes", "y"}
    )

    # Build request-level config dict expected by the legacy build_schedule
    request_config: Dict[str, object] = {
        "transit_coverage_threshold": config.transit_coverage_min,
        "schedule_factor_threshold": config.sched_weights[2],  # schedule weight
        "saa_overlap_threshold": config.saa_overlap_threshold,
        "generate_visibility": generate_visibility,
        "visibility_sun_deg": config.sun_avoidance_deg,
        "visibility_moon_deg": config.moon_avoidance_deg,
        "visibility_earth_deg": config.earth_avoidance_deg,
        "visibility_force": config.force_regenerate,
        "visibility_target_filters": ",".join(config.target_filters) if config.target_filters else None,
    }

    # Prepare extra_inputs for the legacy pipeline; copy existing extras and add GMAT/target base
    request_extra_inputs: Dict[str, Path] = dict(config.extra_inputs or {})
    if config.gmat_ephemeris is not None:
        request_extra_inputs["visibility_gmat"] = Path(config.gmat_ephemeris)
    # Ensure target definition base is propagated if present in extra_inputs
    if "target_definition_base" in request_extra_inputs:
        request_extra_inputs["target_definition_base"] = Path(request_extra_inputs["target_definition_base"])  # type: ignore[index]

    # When generating target manifests from a provided target definition base,
    # write the generated CSV manifests into the run's output data directory so
    # subsequent steps (visibility & calendar generation) can find them there.
    if config.output_dir is not None:
        out_data = Path(config.output_dir) / "data"
        request_extra_inputs.setdefault(
            "primary_target_csv", out_data / "exoplanet_targets.csv"
        )
        request_extra_inputs.setdefault(
            "auxiliary_target_csv", out_data / "auxiliary-standard_targets.csv"
        )
        request_extra_inputs.setdefault(
            "monitoring_target_csv", out_data / "monitoring-standard_targets.csv"
        )
        request_extra_inputs.setdefault(
            "occultation_target_csv", out_data / "occultation-standard_targets.csv"
        )

    request = SchedulerRequest(
        targets_manifest=config.targets_manifest or Path("."),
        window_start=config.window_start,
        window_end=config.window_end,
        output_dir=config.output_dir or Path("output"),
        config={k: v for k, v in request_config.items() if v is not None},
        extra_inputs=request_extra_inputs,
    )
    
    # Call existing build_schedule implementation
    return build_schedule(request)





def _maybe_generate_visibility(
    request: SchedulerRequest,
    paths: SchedulerPaths,
    pandora_start: datetime,
    pandora_stop: datetime,
    primary_target_csv: Path,
    auxiliary_target_csv: Path,
    monitoring_target_csv: Path,
    occultation_target_csv: Path,
) -> None:
    if not _as_bool(request.config.get("generate_visibility"), False):
        return

    # 1. Primary Targets -> data/targets
    primary_config = _build_visibility_config(
        request,
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
        request,
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
        request,
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
        request,
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
    request: SchedulerRequest,
    paths: SchedulerPaths,
    pandora_start: datetime,
    pandora_stop: datetime,
    target_list: Path,
    partner_list: Optional[Path],
    output_subpath: str,
) -> VisibilityConfig:
    config = request.config
    extra_inputs = request.extra_inputs

    default_gmat = paths.data_dir / "Pandora-600km-withoutdrag-20251018.txt"

    gmat_path = extra_inputs.get("visibility_gmat") or _coerce_path(
        config.get("visibility_gmat"),
        default_gmat,
    )

    target_list_path = extra_inputs.get("visibility_target_list") or _coerce_path(
        config.get("visibility_target_list"),
        target_list,
    )

    partner_override = extra_inputs.get("visibility_partner_list")
    partner_value = config.get("visibility_partner_list")

    if partner_override is not None:
        partner_list = partner_override
    elif partner_value is not None:
        if isinstance(partner_value, bool) and not partner_value:
            partner_list = None
        elif isinstance(partner_value, str) and not partner_value.strip():
            partner_list = None
        else:
            default_partner = partner_list if partner_list is not None else (paths.data_dir / "auxiliary-standard_targets.csv")
            partner_list = _coerce_path(partner_value, default_partner)
    # else: use the passed partner_list argument as-is

    output_override = extra_inputs.get("visibility_output_root")
    if output_override is not None:
        output_root = output_override
    else:
        config_root = _coerce_optional_path(config.get("visibility_output_root"))
        if config_root is not None:
            output_root = config_root
        else:
            # Use request output dir + subpath
            output_root = request.output_dir / "data" / output_subpath

    sun_value = config.get("visibility_sun_deg")
    if sun_value is None:
        sun_value = config.get("sun_avoidance_deg")
    sun_deg = _as_float(sun_value, 91.0)

    moon_value = config.get("visibility_moon_deg")
    if moon_value is None:
        moon_value = config.get("moon_avoidance_deg")
    moon_deg = _as_float(moon_value, 25.0)

    earth_value = config.get("visibility_earth_deg")
    if earth_value is None:
        earth_value = config.get("earth_avoidance_deg")
    earth_deg = _as_float(earth_value, 86.0)

    force_flag = _as_bool(config.get("visibility_force"), False)
    target_filters = _coerce_target_filters(config.get("visibility_target_filters"))

    return VisibilityConfig(
        window_start=pandora_start,
        window_end=pandora_stop,
        gmat_ephemeris=gmat_path,
        target_list=target_list_path,
        partner_list=partner_list,
        output_root=output_root,
        sun_avoidance_deg=sun_deg,
        moon_avoidance_deg=moon_deg,
        earth_avoidance_deg=earth_deg,
        force=force_flag,
        target_filters=target_filters,
    )


def _build_config(values: Dict[str, object]) -> SchedulerConfig:
    obs_window_hours = _as_float(values.get("obs_window_hours"), 24.0)
    transit_coverage_min = _as_float(values.get("transit_coverage_min"), 0.4)
    sched_weights_raw = values.get("sched_weights")
    if sched_weights_raw is None:
        sched_weights_raw = (0.8, 0.0, 0.2)
    sched_weights = _coerce_sched_weights(sched_weights_raw)
    min_visibility = _as_float(values.get("min_visibility"), 0.5)
    deprioritization_limit_hours = _as_float(
        values.get("deprioritization_limit_hours"), 48.0
    )
    commissioning_days = _as_int(values.get("commissioning_days"), 0)
    aux_key_value = values.get("aux_key", "sort_by_tdf_priority")
    show_progress = _as_bool(values.get("show_progress"), False)
    std_obs_duration_hours = _as_float(values.get("std_obs_duration_hours"), 0.5)
    std_obs_frequency_days = _as_float(values.get("std_obs_frequency_days"), 3.0)
    if isinstance(aux_key_value, str) and aux_key_value.lower() == "none":
        aux_key = None
    elif aux_key_value is None:
        aux_key = None
    elif isinstance(aux_key_value, str):
        aux_key = aux_key_value
    else:
        raise TypeError("aux_key must be a string or None")

    return SchedulerConfig(
        obs_window=timedelta(hours=obs_window_hours),
        transit_coverage_min=transit_coverage_min,
        sched_weights=sched_weights,
        min_visibility=min_visibility,
        deprioritization_limit_hours=deprioritization_limit_hours,
        commissioning_days=commissioning_days,
        aux_key=aux_key,
        show_progress=show_progress,
        std_obs_duration_hours=std_obs_duration_hours,
        std_obs_frequency_days=std_obs_frequency_days,
    )


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


def _coerce_sched_weights(value: object) -> tuple[float, float, float]:
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
        raise ValueError("sched_weights must provide three numeric components")

    weights = tuple(_as_float(component, 0.0) for component in components)

    if len(weights) != 3:
        raise ValueError(
            "sched_weights must contain exactly three values (coverage, saa, schedule)."
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
