"""Reworked scheduler implementation mirroring the legacy behaviour."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, cast

import numpy as np
import pandas as pd
from astropy.time import Time
from tqdm import tqdm

from pandorascheduler_rework import observation_utils
from pandorascheduler_rework.utils.io import read_csv_cached
from pandorascheduler_rework.utils.string_ops import remove_suffix


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass(frozen=True)
class SchedulerConfig:
    """Configuration knobs for the scheduling loop."""

    obs_window: timedelta
    transit_coverage_min: float
    transit_scheduling_weights: tuple[float, float, float]
    min_visibility: float
    deprioritization_limit_hours: float
    commissioning_days: int = 0
    aux_key: Optional[str] = "sort_by_tdf_priority"
    show_progress: bool = False
    std_obs_duration_hours: float = 0.5
    std_obs_frequency_days: float = 3.0

    def __post_init__(self) -> None:
        if not np.isclose(sum(self.transit_scheduling_weights), 1.0):
            raise ValueError("transit_scheduling_weights must sum to 1.0")


@dataclass(frozen=True)
class SchedulerPaths:
    """Filesystem layout for a run."""

    package_dir: Path
    data_dir: Path
    targets_dir: Path
    aux_targets_dir: Path
    baseline_dir: Path

    @classmethod
    def from_package_root(cls, package_dir: Path) -> "SchedulerPaths":
        data_dir = package_dir / "data"
        return cls(
            package_dir=package_dir,
            data_dir=data_dir,
            targets_dir=data_dir / "targets",
            aux_targets_dir=data_dir / "aux_targets",
            baseline_dir=data_dir / "baseline",
        )


@dataclass
class AuxiliaryObservationStats:
    total_time: timedelta = timedelta()
    last_priority: float = 0.0


@dataclass
class SchedulerState:
    tracker: pd.DataFrame
    all_target_obs_time: Dict[str, timedelta] = field(default_factory=dict)
    non_primary_obs_time: Dict[str, "AuxiliaryObservationStats"] = field(
        default_factory=dict
    )
    last_std_obs: datetime = datetime(2025, 12, 1)


@dataclass(frozen=True)
class SchedulerInputs:
    pandora_start: datetime
    pandora_stop: datetime
    sched_start: datetime
    sched_stop: datetime
    target_list: pd.DataFrame
    paths: SchedulerPaths
    target_definition_files: list[str]
    primary_target_csv: Path
    auxiliary_target_csv: Path
    occultation_target_csv: Path
    output_dir: Path
    tracker_pickle_path: Optional[Path] = None


@dataclass
class SchedulerOutputs:
    schedule: pd.DataFrame
    tracker: pd.DataFrame
    observation_report_path: Optional[Path]
    schedule_path: Optional[Path]
    tracker_csv_path: Optional[Path]
    tracker_pickle_path: Optional[Path]


def run_scheduler(inputs: SchedulerInputs, config: SchedulerConfig) -> SchedulerOutputs:
    """Execute the scheduling loop to mirror the legacy Schedule function."""

    commissioning_offset = timedelta(days=config.commissioning_days)
    pandora_start = inputs.pandora_start + commissioning_offset
    pandora_stop = inputs.pandora_stop
    sched_start = inputs.sched_start + commissioning_offset
    sched_stop = inputs.sched_stop

    state = SchedulerState(
        tracker=_initialize_tracker(
            inputs,
            config,
            pandora_start,
            pandora_stop,
            sched_start,
            sched_stop,
        ),
        all_target_obs_time={},
        non_primary_obs_time={},
        last_std_obs=datetime(2025, 12, 1),
    )

    inputs.output_dir.mkdir(parents=True, exist_ok=True)

    schedule_rows: list[pd.DataFrame] = []

    start = sched_start
    stop = start + config.obs_window

    progress_bar = None
    previous_start = start
    if config.show_progress:
        total_minutes = max((sched_stop - sched_start).total_seconds() / 60.0, 1.0)
        progress_bar = tqdm(
            total=total_minutes,
            desc="Scheduling",
            unit="min",
            dynamic_ncols=True,
        )

    def advance_progress(new_start: datetime) -> None:
        nonlocal previous_start
        if not progress_bar:
            previous_start = new_start
            return
        delta_minutes = (new_start - previous_start).total_seconds() / 60.0
        if delta_minutes > 0:
            remaining = progress_bar.total - progress_bar.n
            progress_bar.update(min(delta_minutes, remaining))
        previous_start = new_start

    too_targets, too_starts, too_stops = _load_too_table(inputs.paths.data_dir)

    # Pre-load all transit windows once
    all_planet_names = state.tracker["Planet Name"].dropna().unique()
    transit_windows = _load_planet_transit_windows(pd.Index(all_planet_names), inputs)

    while stop <= sched_stop:
        state.tracker = state.tracker.sort_values(
            by=["Primary Target", "Transit Priority"],
            ascending=[False, True],
        ).reset_index(drop=True)
        obs_range = pd.date_range(start, stop, freq="min")
        temp_df = observation_utils.check_if_transits_in_obs_window(
            state.tracker,
            pd.DataFrame(
                [],
                columns=[
                    "Planet Name",
                    "Obs Start",
                    "Obs Gap Time",
                    "Transit Coverage",
                    "SAA Overlap",
                    "Schedule Factor",
                    "Transit Factor",
                    "Quality Factor",
                    "Comments",
                ],
            ),
            inputs.target_list,
            start,
            pandora_start,
            pandora_stop,
            sched_start,
            sched_stop,
            obs_range,
            config.obs_window,
            list(config.transit_scheduling_weights),
            config.transit_coverage_min,
        )

        too_result = _handle_targets_of_opportunity(
            start,
            stop,
            obs_range,
            too_targets,
            too_starts,
            too_stops,
            state,
            inputs,
            config,
            transit_windows,
        )
        if too_result is not None:
            too_df, new_start = too_result
            if not too_df.empty:
                schedule_rows.append(too_df)
            start = new_start
            stop = start + config.obs_window
            advance_progress(start)
            continue

        if temp_df.empty:
            aux_df, log_info = _schedule_auxiliary_target(
                start,
                stop,
                config,
                state,
                inputs,
            )
            if not aux_df.empty:
                schedule_rows.append(aux_df)
            logger.info(f"{log_info}; window {start} to {stop}")
            start = stop
            stop = start + config.obs_window
            advance_progress(start)
            continue

        scheduled_visit = _schedule_primary_target(
            temp_df,
            state,
            inputs,
            config,
            start,
            obs_range,
        )
        schedule_rows.append(scheduled_visit)
        start = scheduled_visit["Observation Stop"].iloc[-1]
        stop = start + config.obs_window
        advance_progress(start)

    schedule = (
        pd.concat(schedule_rows, ignore_index=True)
        if schedule_rows
        else pd.DataFrame(
            columns=[
                "Target",
                "Observation Start",
                "Observation Stop",
                "RA",
                "DEC",
                "Transit Coverage",
                "SAA Overlap",
                "Schedule Factor",
                "Quality Factor",
                "Comments",
            ]
        )
    )

    schedule = schedule.sort_values(by=["Observation Start"]).reset_index(drop=True)
    column_order = [
        "Target",
        "Observation Start",
        "Observation Stop",
        "RA",
        "DEC",
        "Transit Coverage",
        "SAA Overlap",
        "Schedule Factor",
        "Quality Factor",
        "Comments",
    ]
    schedule = schedule.reindex(columns=column_order)

    observation_report_path = _write_observation_report(
        inputs,
        state,
        pandora_start,
    )
    schedule_path, tracker_csv_path, tracker_pickle_path = _persist_outputs(
        schedule,
        state.tracker,
        inputs,
        config,
        pandora_start,
        pandora_stop,
    )

    if progress_bar:
        advance_progress(sched_stop)
        progress_bar.close()

    return SchedulerOutputs(
        schedule=schedule,
        tracker=state.tracker,
        observation_report_path=observation_report_path,
        schedule_path=schedule_path,
        tracker_csv_path=tracker_csv_path,
        tracker_pickle_path=tracker_pickle_path,
    )


def _initialize_tracker(
    inputs: SchedulerInputs,
    config: SchedulerConfig,
    pandora_start: datetime,
    pandora_stop: datetime,
    sched_start: datetime,
    sched_stop: datetime,
) -> pd.DataFrame:
    target_list = inputs.target_list.reset_index(drop=True)

    tracker = pd.DataFrame(
        {
            "Planet Name": target_list["Planet Name"].to_numpy(),
            "Primary Target": target_list["Primary Target"].to_numpy(),
            "RA": target_list["RA"].to_numpy(),
            "DEC": target_list["DEC"].to_numpy(),
            "Transits Needed": target_list[
                "Number of Transits to Capture"
            ].to_numpy(),
        }
    )
    tracker["Transits Acquired"] = np.zeros(len(target_list), dtype=float)

    archive_path = inputs.paths.data_dir / "Pandora_archive.csv"
    if archive_path.exists():
        try:
            archive = pd.read_csv(archive_path)
        except Exception:
            archive = None
        if archive is not None:
            for _, row in archive.iterrows():
                mask = tracker["Planet Name"] == row["Target"]
                tracker.loc[mask, "Transits Needed"] -= 1
                tracker.loc[mask, "Transits Acquired"] += 1

    transits_left_lifetime: list[int] = []
    transits_left_schedule: list[int] = []

    for _, row in target_list.iterrows():
        planet_name = str(row["Planet Name"])
        star_name = str(row["Star Name"])

        visibility_path = observation_utils.build_visibility_path(
            inputs.paths.targets_dir, star_name, planet_name
        )
        planet_data = None
        try:
            planet_data = read_csv_cached(str(visibility_path))
        except Exception:
            planet_data = None

        if planet_data is None:
            raise FileNotFoundError(
                f"Visibility file missing or unreadable for planet {planet_name} (expected at {visibility_path})"
            )

        planet_data = planet_data.drop(
            planet_data.index[
                planet_data["Transit_Coverage"] < config.transit_coverage_min
            ]
        ).reset_index(drop=True)

        # If no transits remain after filtering, skip this planet
        if planet_data.empty or len(planet_data) == 0:
            transits_left_lifetime.append(0)
            transits_left_schedule.append(0)
            continue

        # Use pre-converted datetime if available (performance optimization)
        if "Transit_Start_UTC" in planet_data.columns:
            start_transits = pd.to_datetime(planet_data["Transit_Start_UTC"]).to_numpy()
        else:
            # Fallback to MJD conversion for backward compatibility
            start_transits = Time(
                planet_data["Transit_Start"], format="mjd", scale="utc"
            ).to_value("datetime")
        
        if "Transit_Stop_UTC" in planet_data.columns:
            end_transits = pd.to_datetime(planet_data["Transit_Stop_UTC"]).to_numpy()
        else:
            # Fallback to MJD conversion for backward compatibility
            end_transits = Time(
                planet_data["Transit_Stop"], format="mjd", scale="utc"
            ).to_value("datetime")

        lifetime_mask = (pandora_start <= start_transits) & (
            end_transits <= pandora_stop
        )
        schedule_mask = (sched_start <= start_transits) & (end_transits <= sched_stop)

        transits_left_lifetime.append(int(np.count_nonzero(lifetime_mask)))
        transits_left_schedule.append(int(np.count_nonzero(schedule_mask)))

    tracker["Transits Left in Lifetime"] = transits_left_lifetime
    tracker["Transits Left in Schedule"] = transits_left_schedule
    tracker["Transit Priority"] = (
        tracker["Transits Left in Lifetime"] - tracker["Transits Needed"]
    )

    return tracker


def _write_observation_report(
    inputs: SchedulerInputs, state: SchedulerState, pandora_start: datetime
) -> Path:
    report_name = f"Observation_Time_Report_{pandora_start}.csv"
    report_path = inputs.output_dir / report_name
    observation_utils.save_observation_time_report(
        state.all_target_obs_time,
        inputs.target_list,
        str(report_path),
    )
    return report_path


def _persist_outputs(
    schedule: pd.DataFrame,
    tracker: pd.DataFrame,
    inputs: SchedulerInputs,
    config: SchedulerConfig,
    pandora_start: datetime,
    pandora_stop: datetime,
) -> tuple[Path, Path, Path]:
    schedule_name = (
        f"Pandora_Schedule_{config.transit_scheduling_weights[0]}_"
        f"{config.transit_scheduling_weights[1]}_{config.transit_scheduling_weights[2]}_"
        f"{pandora_start.strftime('%Y-%m-%d')}_to_{pandora_stop.strftime('%Y-%m-%d')}.csv"
    )
    schedule_path = inputs.output_dir / schedule_name
    schedule.to_csv(schedule_path, index=False)

    tracker_csv_path = inputs.output_dir / "tracker.csv"
    tracker.to_csv(tracker_csv_path, index=False)

    tracker_pickle_path = inputs.tracker_pickle_path or (
        inputs.output_dir
        / f"Tracker_{pandora_start.strftime('%Y-%m-%d')}_to_{pandora_stop.strftime('%Y-%m-%d')}.pkl"
    )
    with tracker_pickle_path.open("wb") as handle:
        pickle.dump(tracker, handle)

    return schedule_path, tracker_csv_path, tracker_pickle_path


def _load_too_table(data_dir: Path) -> tuple[list[str], list[datetime], list[datetime]]:
    too_path = data_dir / "ToO_list.csv"
    if not too_path.exists():
        return [], [], []

    table = pd.read_csv(too_path)
    return (
        table["Target"].tolist(),
        [
            datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            for value in table["Obs Window Start"]
        ],
        [
            datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            for value in table["Obs Window Stop"]
        ],
    )


def _load_planet_transit_windows(
    planet_names: pd.Index,
    inputs: SchedulerInputs,
) -> dict[str, tuple[datetime, datetime]]:
    """Batch-load transit windows for multiple planets to avoid repeated CSV reads."""
    transit_windows = {}
    
    for planet_name in planet_names:
        planet_str = str(planet_name)
        star_name = remove_suffix(planet_str)
        try:
            planet_visibility = read_csv_cached(
                str(observation_utils.build_visibility_path(
                    inputs.paths.targets_dir, star_name, planet_str
                ))
            )
        except FileNotFoundError:
            continue

        if planet_visibility is None or planet_visibility.empty or len(planet_visibility) == 0:
            continue

        start_transit = Time(
            planet_visibility["Transit_Start"].iloc[-1], format="mjd", scale="utc"
        ).to_datetime()
        end_transit = Time(
            planet_visibility["Transit_Stop"].iloc[-1], format="mjd", scale="utc"
        ).to_datetime()

        start_transit = start_transit.replace(second=0, microsecond=0)
        end_transit = end_transit.replace(second=0, microsecond=0)

        transit_windows[str(planet_name)] = (start_transit, end_transit)
    
    return transit_windows


def _handle_targets_of_opportunity(
    start: datetime,
    stop: datetime,
    obs_range: pd.DatetimeIndex,
    targets: list[str],
    starts: list[datetime],
    stops: list[datetime],
    state: SchedulerState,
    inputs: SchedulerInputs,
    config: SchedulerConfig,
    transit_windows: dict[str, tuple[datetime, datetime]],
) -> Optional[tuple[pd.DataFrame, datetime]]:
    if not targets:
        return None

    overlap = obs_range.intersection(pd.DatetimeIndex(starts))
    if len(overlap) == 0:
        return None

    first_overlap = overlap[0].to_pydatetime()
    idx = starts.index(first_overlap)
    obs_start = starts[idx]
    obs_stop = stops[idx]
    target_name = targets[idx]

    logger.info("Attempting to schedule Target of Opportunity")

    tracker = state.tracker
    positive_needed = tracker["Transits Needed"] > 0
    transit_ratio = pd.Series(np.inf, index=tracker.index)
    transit_ratio.loc[positive_needed] = (
        tracker.loc[positive_needed, "Transits Left in Lifetime"]
        / tracker.loc[positive_needed, "Transits Needed"]
    )
    critical_planets = tracker.loc[positive_needed & (transit_ratio <= 1)].copy()

    schedule_parts: list[pd.DataFrame] = []
    forced_observation = False

    # Batch-load all transit windows once
    # all_active_names = tracker.loc[positive_needed].index
    # transit_windows = _load_planet_transit_windows(all_active_names, inputs)

    for planet_name in critical_planets.index:
        if planet_name not in transit_windows:
            continue

        start_transit, end_transit = transit_windows[planet_name]

        early_start = end_transit - timedelta(hours=20)
        late_start = start_transit - timedelta(hours=4)

        start_range = pd.date_range(early_start, late_start, freq="min")
        overlap_times = obs_range.intersection(start_range)

        if len(overlap_times) == 0:
            continue

        forced_observation = True
        forced_start = overlap_times[0].to_pydatetime()

        if obs_range[0].to_pydatetime() < forced_start:
            free_df = pd.DataFrame(
                [
                    [
                        "FREE PRE-TOO, REPLACE WITH AUX",
                        obs_range[0].to_pydatetime(),
                        forced_start,
                    ]
                ],
                columns=["Target", "Observation Start", "Observation Stop"],
            )
            schedule_parts.append(free_df)

        forced_df = pd.DataFrame(
            [
                [
                    planet_name,
                    forced_start,
                    obs_stop,
                    f"{planet_name} forced over ToO due to transit factor <=1",
                ]
            ],
            columns=["Target", "Observation Start", "Observation Stop", "Comments"],
        )
        schedule_parts.append(forced_df)
        logger.info(
            "Forced observation of %s over ToO due to critical transit factor",
            planet_name,
        )
        break

    if forced_observation:
        combined = (
            pd.concat(schedule_parts, ignore_index=True)
            if schedule_parts
            else pd.DataFrame()
        )
        return combined, obs_stop

    tf_warning_messages: list[str] = []
    active_planets = tracker.loc[positive_needed]
    
    # Reuse the already-loaded transit windows
    for planet_name in active_planets.index:
        if planet_name not in transit_windows:
            continue

        tf_ratio = transit_ratio.loc[planet_name]
        start_transit, end_transit = transit_windows[planet_name]

        early_start = end_transit - timedelta(hours=20)
        late_start = start_transit - timedelta(hours=4)

        start_range = pd.date_range(early_start, late_start, freq="min")
        overlap_times = obs_range.intersection(start_range)

        if len(overlap_times) > 0 and tf_ratio > 1:
            tf_warning_messages.append(
                f"Warning: {planet_name} has MTRM > 1 and is transiting during ToO."
            )

    if obs_range[0].to_pydatetime() < obs_start:
        free_df = pd.DataFrame(
            [
                [
                    "FREE PRE-TOO, REPLACE WITH AUX",
                    obs_range[0].to_pydatetime(),
                    obs_start,
                ]
            ],
            columns=["Target", "Observation Start", "Observation Stop"],
        )
        schedule_parts.append(free_df)

    too_df = pd.DataFrame(
        [
            [
                target_name,
                obs_start,
                obs_stop,
                " ".join(tf_warning_messages).strip(),
            ]
        ],
        columns=["Target", "Observation Start", "Observation Stop", "Comments"],
    )
    schedule_parts.append(too_df)

    logger.info(f"Scheduled Target of Opportunity: {target_name}")
    for message in tf_warning_messages:
        logger.warning(message)

    combined = pd.concat(schedule_parts, ignore_index=True)
    return combined, obs_stop


def _schedule_auxiliary_target(
    start: datetime,
    stop: datetime,
    config: SchedulerConfig,
    state: SchedulerState,
    inputs: SchedulerInputs,
) -> tuple[pd.DataFrame, str]:
    obs_range = pd.date_range(start, stop, freq="min")
    active_start = start
    scheduled_rows: list[list] = []
    row_columns = ["Target", "Observation Start", "Observation Stop", "RA", "DEC"]

    obs_std_duration = timedelta(hours=config.std_obs_duration_hours)
    if (
        active_start - state.last_std_obs > timedelta(days=config.std_obs_frequency_days)
        and active_start + obs_std_duration < stop
    ):
        std_path = inputs.paths.data_dir / "monitoring-standard_targets.csv"
        std_df = read_csv_cached(str(std_path))
        if std_df is not None:
            std_df = std_df.sort_values(
                "Priority", ascending=False, ignore_index=True
            )
            std_records = std_df.to_dict(orient="records")
        else:
            std_records = []

        std_candidate: Optional[tuple[str, float, float, float]] = None
        for std_row in std_records:
            std_name = str(std_row["Star Name"])
            vis_file = observation_utils.build_star_visibility_path(
                inputs.paths.aux_targets_dir, std_name
            )
            try:
                vis = read_csv_cached(str(vis_file))
            except FileNotFoundError:
                continue
            
            
            if vis is None or vis.empty or len(vis) == 0:
                continue

            # Use pre-converted datetime if available (performance optimization)
            if "Time_UTC" in vis.columns:
                vis_times = pd.to_datetime(vis["Time_UTC"])
            else:
                # Fallback to MJD conversion for backward compatibility
                vis_times = Time(
                    vis["Time(MJD_UTC)"].to_numpy(), format="mjd", scale="utc"
                ).to_datetime()
                vis_times = pd.to_datetime(vis_times)
            
            mask = (vis_times >= active_start) & (
                vis_times <= active_start + obs_std_duration
            )
            vis_filtered = vis.loc[mask]

            if not vis_filtered.empty and vis_filtered["Visible"].all():
                std_candidate = (
                    std_name,
                    float(std_row["RA"]),
                    float(std_row["DEC"]),
                    float(std_row["Priority"]),
                )
                logger.info(f"{std_name} scheduled for STD observations with full visibility")
                break

        if std_candidate is None:
            std_candidate = (
                "WARNING: no visible standard star",
                float("nan"),
                float("nan"),
                1.0,
            )
            logger.warning(
                "No visible standard star between %s and %s",
                active_start,
                active_start + obs_std_duration,
            )

        std_name, std_ra, std_dec, priority_std = std_candidate
        std_end = active_start + obs_std_duration
        scheduled_rows.append(
            [f"{std_name} STD", active_start, std_end, std_ra, std_dec]
        )
        active_start = std_end
        state.last_std_obs = active_start

        stats = state.non_primary_obs_time.setdefault(
            "STD", AuxiliaryObservationStats()
        )
        stats.total_time += obs_std_duration
        stats.last_priority = priority_std

    if config.aux_key is None:
        scheduled_rows.append(
            ["Free Time", active_start, stop, float("nan"), float("nan")]
        )
        result = pd.DataFrame(scheduled_rows, columns=row_columns)
        for record in result.to_dict(orient="records"):
            target_label = str(record["Target"])
            if target_label == "Free Time":
                continue
            duration = record["Observation Stop"] - record["Observation Start"]
            if isinstance(duration, pd.Timedelta):
                duration = duration.to_pytimedelta()
            state.all_target_obs_time[target_label] = (
                state.all_target_obs_time.get(target_label, timedelta()) + duration
            )
        return result, "Free time, no observation scheduled."

    selected_row: Optional[list] = None
    priority_val = 0.0
    priority_baseline = 0.0
    log_info = "No fuly or partially visible non-primary targets, Free Time..."

    deprioritization_limit = timedelta(hours=config.deprioritization_limit_hours)
    non_primary_priorities = {
        name: stats.last_priority
        for name, stats in state.non_primary_obs_time.items()
        if not name.endswith("STD")
    }

    for target_def in inputs.target_definition_files[1:]:
        aux_csv = inputs.paths.data_dir / f"{target_def}_targets.csv"
        if not aux_csv.exists():
            continue

        aux_targets = read_csv_cached(str(aux_csv))
        if aux_targets is None:
            continue
        aux_targets = aux_targets.reset_index(drop=True)

        mask = aux_targets["Star Name"].isin(non_primary_priorities.keys())
        if mask.any():
            mapped_priorities = aux_targets.loc[mask, "Star Name"].apply(
                lambda value: non_primary_priorities.get(str(value), np.nan)
            )
            aux_targets.loc[mask, "Priority"] = mapped_priorities

        if config.aux_key in {"sort_by_tdf_priority", "closest"}:
            aux_targets = aux_targets.sort_values(
                "Priority", ascending=False, ignore_index=True
            )
        names = aux_targets["Star Name"].tolist()
        ras = aux_targets["RA"].tolist()
        decs = aux_targets["DEC"].tolist()
        priorities = aux_targets["Priority"].tolist()

        vis_all: list[int] = []
        vis_any: list[int] = []
        vis_percentages: list[float] = []

        for idx, name in enumerate(names):
            vis_file = observation_utils.build_star_visibility_path(
                inputs.paths.aux_targets_dir, name
            )
            try:
                vis = read_csv_cached(str(vis_file))
            except FileNotFoundError:
                continue
            
            if vis is None or vis.empty or len(vis) == 0:
                continue

            # Use pre-converted datetime if available (performance optimization)
            if "Time_UTC" in vis.columns:
                vis_times = pd.to_datetime(vis["Time_UTC"])
            else:
                # Fallback to MJD conversion for backward compatibility
                vis_times = Time(
                    vis["Time(MJD_UTC)"].to_numpy(), format="mjd", scale="utc"
                ).to_datetime()
                vis_times = pd.to_datetime(vis_times)
            
            mask = (vis_times >= active_start) & (vis_times <= stop)
            vis_filtered = vis.loc[mask]

            if not vis_filtered.empty and vis_filtered["Visible"].all():
                vis_all.append(idx)
                break
            if not vis_filtered.empty and vis_filtered["Visible"].any():
                vis_any.append(idx)
                visibility = 100 * (vis_filtered["Visible"].sum() / len(vis_filtered))
                vis_percentages.append(visibility)

        if vis_all:
            if config.aux_key in {"sort_by_tdf_priority", "closest"}:
                chosen_idx = vis_all[0]
            else:
                chosen_idx = int(np.random.randint(0, len(vis_all)))
            name = names[chosen_idx]
            ra_val = ras[chosen_idx]
            dec_val = decs[chosen_idx]
            priority_val = priorities[chosen_idx]
            log_info = (
                f"{name} scheduled for non-primary observation with full visibility."
            )
            scheduled_rows.append([name, active_start, stop, ra_val, dec_val])
            logger.info(
                f"{name} scheduled for non-primary observations with full visibility from {target_def}"
            )
            selected_row = scheduled_rows[-1]
            priority_baseline = priorities[-1] if priorities else priority_val
            break

        if vis_any:
            chosen_idx = int(np.asarray(vis_percentages).argmax())
            best_visibility = vis_percentages[chosen_idx]
            if best_visibility >= 100 * config.min_visibility:
                idx_value = vis_any[chosen_idx]
                name = names[idx_value]
                ra_val = ras[idx_value]
                dec_val = decs[idx_value]
                priority_val = priorities[idx_value]
                log_info = f"No non-primary target with full visibility; {name} scheduled for non-primary observations with {best_visibility:.2f}% visibility from {target_def}."
                scheduled_rows.append([name, active_start, stop, ra_val, dec_val])
                logger.info(
                    f"No non-primary target with full visibility; {name} scheduled at {best_visibility:.2f}% visibility from {target_def}"
                )
                selected_row = scheduled_rows[-1]
                priority_baseline = priorities[-1] if priorities else priority_val
                break
            logger.warning(
                "No non-primary target with visibility greater than %.2f%% from %s",
                100 * config.min_visibility,
                target_def,
            )

    if selected_row is None:
        scheduled_rows.append(
            ["Free Time", active_start, stop, float("nan"), float("nan")]
        )
    else:
        name = selected_row[0]
        stats = state.non_primary_obs_time.setdefault(name, AuxiliaryObservationStats())
        stats.total_time += stop - active_start
        stats.last_priority = priority_val

        if stats.total_time > deprioritization_limit:
            logger.warning("Deprioritizing %s due to accumulated auxiliary time", name)
            baseline = priority_baseline if priority_baseline else priority_val
            stats.last_priority = 0.95 * float(baseline)

    result = pd.DataFrame(scheduled_rows, columns=row_columns)
    # Update all_target_obs_time
    for record in result.to_dict(orient="records"):
        target_label = str(record["Target"])
        if target_label == "Free Time":
            continue
        duration = record["Observation Stop"] - record["Observation Start"]
        if isinstance(duration, pd.Timedelta):
            duration = duration.to_pytimedelta()
        state.all_target_obs_time[target_label] = (
            state.all_target_obs_time.get(target_label, timedelta()) + duration
        )
    
    return result, log_info


def _schedule_primary_target(
    temp_df: pd.DataFrame,
    state: SchedulerState,
    inputs: SchedulerInputs,
    config: SchedulerConfig,
    start: datetime,
    obs_range: pd.DatetimeIndex,
) -> pd.DataFrame:
    if (temp_df["Transit Factor"] <= 2).any():
        ranked = temp_df.sort_values(by=["Transit Factor"]).reset_index(drop=True)
    else:
        ranked = temp_df.sort_values(
            by=["Quality Factor", "Transit Factor"],
            ascending=[False, True],
        ).reset_index(drop=True)

    first_row = ranked.iloc[0]
    planet_name = str(first_row["Planet Name"])
    ra_value = first_row["RA"]
    dec_value = first_row["DEC"]
    obs_start = pd.Timestamp(first_row["Obs Start"]).to_pydatetime()
    obs_stop = obs_start + config.obs_window
    trans_cover = first_row["Transit Coverage"]
    saa_cover = first_row["SAA Overlap"]
    s_factor = first_row["Schedule Factor"]
    q_factor = first_row["Quality Factor"]

    dfs: list[pd.DataFrame] = []

    if obs_range[0].to_pydatetime() < obs_start:
        aux_df, aux_log = _schedule_auxiliary_target(
            start,
            obs_start,
            config,
            state,
            inputs,
        )
        if not aux_df.empty:
            dfs.append(aux_df)
        logger.info(f"{aux_log}; window {start} to {obs_start}")

    main_schedule = pd.DataFrame(
        [
            [
                planet_name,
                obs_start,
                obs_stop,
                ra_value,
                dec_value,
                trans_cover,
                saa_cover,
                s_factor,
                q_factor,
                np.nan,
            ]
        ],
        columns=[
            "Target",
            "Observation Start",
            "Observation Stop",
            "RA",
            "DEC",
            "Transit Coverage",
            "SAA Overlap",
            "Schedule Factor",
            "Quality Factor",
            "Comments",
        ],
    )
    dfs.append(main_schedule)

    state.all_target_obs_time[planet_name] = state.all_target_obs_time.get(
        planet_name, timedelta()
    ) + (obs_stop - obs_start)

    tracker_mask = state.tracker["Planet Name"] == planet_name
    state.tracker.loc[tracker_mask, "Transits Needed"] -= 1
    state.tracker.loc[tracker_mask, "Transits Acquired"] += 1
    state.tracker.loc[tracker_mask, "Transits Needed"] = state.tracker.loc[
        tracker_mask, "Transits Needed"
    ].clip(lower=0)
    state.tracker.loc[tracker_mask, "Transit Priority"] = (
        state.tracker.loc[tracker_mask, "Transits Left in Lifetime"]
        - state.tracker.loc[tracker_mask, "Transits Needed"]
    )

    acquired_value = state.tracker.loc[tracker_mask, "Transits Acquired"].iloc[0]
    acquired = int(cast(float, acquired_value))
    logger.info(
        f"Scheduled transit {acquired} of {planet_name}. Transit coverage: {trans_cover:.2f}",
    )

    return pd.concat(dfs, ignore_index=True)
