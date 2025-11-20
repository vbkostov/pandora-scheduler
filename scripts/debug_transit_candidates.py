from __future__ import annotations

from datetime import datetime, timedelta
import importlib.util
from pathlib import Path
from typing import Iterable

import pandas as pd
from astropy.time import Time

from pandorascheduler_rework import observation_utils as rework_helper
from pandorascheduler_rework.scheduler import (
    SchedulerConfig,
    SchedulerInputs,
    SchedulerPaths,
    _initialize_tracker,
)


def main() -> None:
    base_start = datetime.fromisoformat("2026-02-05 00:00:00")
    window_days = 7
    window_start = base_start
    window_end = base_start + timedelta(days=window_days)

    pandora_start = window_start
    pandora_stop = window_end

    repo_root = Path(__file__).resolve().parents[1]
    legacy_package_dir = repo_root / "src" / "pandorascheduler"
    paths = SchedulerPaths.from_package_root(legacy_package_dir)

    target_list = pd.read_csv(paths.data_dir / "exoplanet_targets.csv")

    legacy_helper_spec = importlib.util.spec_from_file_location(
        "legacy_helper_codes",
        legacy_package_dir / "observation_utils.py",
    )
    if legacy_helper_spec is None or legacy_helper_spec.loader is None:
        raise RuntimeError("Unable to load legacy helper_codes module")

    legacy_helper = importlib.util.module_from_spec(legacy_helper_spec)
    legacy_helper_spec.loader.exec_module(legacy_helper)
    legacy_check = legacy_helper.check_if_transits_in_obs_window

    inputs = SchedulerInputs(
        pandora_start=pandora_start,
        pandora_stop=pandora_stop,
        sched_start=window_start,
        sched_stop=window_end,
        target_list=target_list,
        paths=paths,
        target_definition_files=[
            "exoplanet",
            "auxiliary-standard",
            "monitoring-standard",
            "occultation-standard",
        ],
        primary_target_csv=paths.data_dir / "exoplanet_targets.csv",
        auxiliary_target_csv=paths.data_dir / "auxiliary-standard_targets.csv",
        occultation_target_csv=paths.data_dir / "occultation-standard_targets.csv",
        output_dir=repo_root / "comparison_outputs" / "scratch",
    )

    config = SchedulerConfig(
        obs_window=timedelta(hours=24),
        transit_coverage_min=0.4,
        sched_weights=(0.8, 0.0, 0.2),
        min_visibility=0.5,
        deprioritization_limit_hours=48.0,
        commissioning_days=0,
        aux_key="sort_by_tdf_priority",
        show_progress=False,
    )

    tracker = _initialize_tracker(
        inputs,
        config,
        pandora_start,
        pandora_stop,
        window_start,
        window_end,
    )

    print("DATA_ROOTS:")
    for root in rework_helper.DATA_ROOTS:
        print("  ", root)

    template_columns = [
        "Planet Name",
        "Obs Start",
        "Obs Gap Time",
        "Transit Coverage",
        "SAA Overlap",
        "Schedule Factor",
        "Transit Factor",
        "Quality Factor",
        "Comments",
    ]
    temp_df = pd.DataFrame(columns=template_columns)

    obs_range = pd.date_range(window_start, window_start + config.obs_window, freq="min")

    rework_tracker = tracker.copy(deep=True)
    rework_candidates = rework_helper.check_if_transits_in_obs_window(
        rework_tracker,
        temp_df.copy(),
        target_list,
        window_start,
        pandora_start,
        pandora_stop,
        window_start,
        window_end,
        obs_range,
        config.obs_window,
        list(config.sched_weights),
        config.transit_coverage_min,
    )

    legacy_tracker = tracker.copy(deep=True)
    legacy_candidates = legacy_check(
        legacy_tracker,
        temp_df.copy(),
        target_list,
        window_start,
        pandora_start,
        pandora_stop,
        window_start,
        window_end,
        obs_range,
        config.obs_window,
        list(config.sched_weights),
        config.transit_coverage_min,
    )

    print("Rework candidates:", len(rework_candidates))
    if not rework_candidates.empty:
        print(rework_candidates.head())

    print("Legacy candidates:", len(legacy_candidates))
    if not legacy_candidates.empty:
        print(legacy_candidates.head())

    if not legacy_candidates.empty:
        debug_planets = legacy_candidates["Planet Name"].unique()
        _diagnose_planets(
            debug_planets,
            tracker,
            target_list,
            window_start,
            window_end,
            pandora_start,
            pandora_stop,
            obs_range,
            config,
        )


def _diagnose_planets(
    planets: Iterable[str],
    tracker: pd.DataFrame,
    target_list: pd.DataFrame,
    sched_start: datetime,
    sched_stop: datetime,
    pandora_start: datetime,
    pandora_stop: datetime,
    obs_range: pd.DatetimeIndex,
    config: SchedulerConfig,
) -> None:
    for planet_name in planets:
        mask = tracker["Planet Name"] == planet_name
        if not mask.any():
            continue

        tracker_row = tracker.loc[mask].iloc[0]
        star_row = target_list.loc[target_list["Planet Name"] == planet_name]
        if star_row.empty:
            continue
        star_name = star_row["Star Name"].iloc[0]
        visibility_file = rework_helper._planet_visibility_file(star_name, planet_name)  # type: ignore[attr-defined]
        if visibility_file is None:
            print(f"{planet_name}: visibility file missing")
            continue

        planet_data = pd.read_csv(visibility_file)
        planet_data = planet_data.drop(
            planet_data.index[
                planet_data["Transit_Coverage"] < config.transit_coverage_min
            ]
        ).reset_index(drop=True)
        start_times = Time(
            planet_data["Transit_Start"].to_numpy(), format="mjd", scale="utc"
        ).to_value("datetime")
        stop_times = Time(
            planet_data["Transit_Stop"].to_numpy(), format="mjd", scale="utc"
        ).to_value("datetime")

        start_series = pd.Series(list(start_times))
        stop_series = pd.Series(list(stop_times))
        valid_mask = start_series >= sched_start
        start_series = start_series.loc[valid_mask].reset_index(drop=True)
        stop_series = stop_series.loc[valid_mask].reset_index(drop=True)
        if start_series.empty:
            print(f"{planet_name}: no starts after window start")
            continue

        start_series = start_series.dt.floor("min")
        stop_series = stop_series.dt.floor("min")
        early_start = stop_series - timedelta(hours=20)
        late_start = start_series - timedelta(hours=4)

        print(f"{planet_name}: evaluating {len(early_start)} windows")
        for idx, (window_start, window_stop) in enumerate(zip(early_start, late_start)):
            start_rng = pd.date_range(window_start, window_stop, freq="min")
            overlap = obs_range.intersection(start_rng)
            print(
                f"  window {idx}: start {window_start}, stop {window_stop}, "
                f"range len {len(start_rng)}, overlap {len(overlap)}"
            )
            if not overlap.empty:
                obs_start = overlap[0]
                gap = obs_start - obs_range[0]
                print(f"    overlap starts at {obs_start} (gap {gap})")


if __name__ == "__main__":
    main()
