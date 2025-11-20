"""High-level entry points for the reworked scheduler pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from pandorascheduler_rework import observation_utils as rework_helper
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

    if not request.targets_manifest.exists():
        raise FileNotFoundError(
            f"Targets manifest does not exist: {request.targets_manifest}"
        )

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

    _maybe_generate_visibility(
        request,
        paths,
        pandora_start,
        pandora_stop,
        primary_target_csv,
        auxiliary_target_csv,
    )

    target_list = pd.read_csv(primary_target_csv)

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


def _maybe_generate_visibility(
    request: SchedulerRequest,
    paths: SchedulerPaths,
    pandora_start: datetime,
    pandora_stop: datetime,
    primary_target_csv: Path,
    auxiliary_target_csv: Path,
) -> None:
    if not _as_bool(request.config.get("generate_visibility"), False):
        return

    visibility_config = _build_visibility_config(
        request,
        paths,
        pandora_start,
        pandora_stop,
        primary_target_csv,
        auxiliary_target_csv,
    )

    LOGGER.info(
        "Generating visibility artifacts for %s", visibility_config.target_list.name
    )
    build_visibility_catalog(visibility_config)


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

def _build_visibility_config(
    request: SchedulerRequest,
    paths: SchedulerPaths,
    pandora_start: datetime,
    pandora_stop: datetime,
    primary_target_csv: Path,
    auxiliary_target_csv: Path,
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
        primary_target_csv,
    )

    partner_override = extra_inputs.get("visibility_partner_list")
    partner_value = config.get("visibility_partner_list")
    partner_list: Optional[Path]
    if partner_override is not None:
        partner_list = partner_override
    elif isinstance(partner_value, bool) and not partner_value:
        partner_list = None
    elif partner_value is None:
        partner_list = auxiliary_target_csv if auxiliary_target_csv.exists() else None
    else:
        partner_str = str(partner_value).strip()
        partner_list = (
            None
            if not partner_str
            else _coerce_path(partner_value, auxiliary_target_csv)
        )

    output_override = extra_inputs.get("visibility_output_root")
    if output_override is not None:
        output_root = output_override
    else:
        output_root = _coerce_optional_path(config.get("visibility_output_root"))

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
    prefer_catalog = _as_bool(
        config.get("visibility_prefer_catalog_coordinates"), False
    )

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
        prefer_catalog_coordinates=prefer_catalog,
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
