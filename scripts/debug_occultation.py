#!/usr/bin/env python3
"""Debug occultation target selection for a specific schedule row."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from astropy.time import Time

from pandorascheduler_rework.xml_builder import (
    ScienceCalendarConfig,
    ScienceCalendarInputs,
    _ScienceCalendarBuilder,
    _extract_visibility_segment,
    _normalise_target_name,
    _occultation_windows,
    _parse_datetime,
    _read_visibility,
    _visibility_change_indices,
)


@dataclass(frozen=True)
class OccultationInputs:
    schedule_index: int
    target_name: str
    star_name: str
    visit_start: pd.Timestamp
    visit_stop: pd.Timestamp
    oc_start_times: Tuple[pd.Timestamp, ...]
    oc_stop_times: Tuple[pd.Timestamp, ...]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare legacy vs rework occultation helpers")
    parser.add_argument(
        "--schedule",
        type=Path,
        default=Path("comparison_outputs/rework/Pandora_Schedule_0.8_0.0_0.2_2026-02-05_to_2026-02-12.csv"),
        help="CSV schedule to inspect (defaults to the rework comparison output)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("src/pandorascheduler/data"),
        help="Directory containing target visibility files (defaults to legacy data bundle)",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        required=True,
        help="Row index within the schedule CSV to analyse",
    )
    parser.add_argument(
        "--legacy-module",
        type=Path,
        default=Path("src/pandorascheduler/sched2xml_WIP.py"),
        help="Path to the legacy sched2xml implementation",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def load_legacy_module(path: Path, working_dir: Path, schedule_source: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Legacy script not found: {path}")

    legacy_root = working_dir / "legacy"
    shutil.copytree(path.parent, legacy_root)

    data_dir = legacy_root / "data"
    default_schedule_name = "Pandora_Schedule_0.8_0.0_0.2_2026-02-05_to_2027-02-05.csv"
    default_schedule_path = data_dir / default_schedule_name
    shutil.copy2(schedule_source, default_schedule_path)

    spec = importlib.util.spec_from_file_location("legacy_sched2xml", legacy_root / path.name)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {path}")

    module = importlib.util.module_from_spec(spec)
    module_dir = str(legacy_root)
    sys.path.insert(0, module_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(module_dir)
    return module, legacy_root


def gather_occultation_inputs(schedule_path: Path, data_dir: Path, row_index: int) -> OccultationInputs:
    inputs = ScienceCalendarInputs(schedule_csv=schedule_path, data_dir=data_dir)
    builder = _ScienceCalendarBuilder(inputs, ScienceCalendarConfig(visit_limit=None))
    row = builder.schedule.iloc[row_index]

    start = _parse_datetime(row["Observation Start"])
    stop = _parse_datetime(row["Observation Stop"])
    if start is None or stop is None:
        raise ValueError("Unable to parse observation window for selected row")

    target_name, star_name = _normalise_target_name(str(row["Target"]))

    vis_frame = _read_visibility(builder.data_dir / "targets" / star_name, star_name)
    if vis_frame is None:
        raise FileNotFoundError(f"Visibility file missing for {star_name}")

    times, flags = _extract_visibility_segment(
        vis_frame,
        start,
        stop,
        builder.config.min_sequence_minutes,
    )
    changes = _visibility_change_indices(flags)
    oc_starts, oc_stops, _ = _occultation_windows(times, flags, changes)

    if not oc_starts:
        raise RuntimeError("No occultation windows detected for this row")

    return OccultationInputs(
        schedule_index=row_index,
        target_name=target_name,
        star_name=star_name,
        visit_start=pd.Timestamp(start),
        visit_stop=pd.Timestamp(stop),
        oc_start_times=tuple(pd.Timestamp(value) for value in oc_starts),
        oc_stop_times=tuple(pd.Timestamp(value) for value in oc_stops),
    )


def run_rework_helper(inputs: OccultationInputs, data_dir: Path):
    from pandorascheduler_rework import observation_utils

    occ_list = pd.read_csv(data_dir / "occultation-standard_targets.csv")
    occ_df = pd.DataFrame(
        {
            "Target": ["" for _ in inputs.oc_start_times],
            "start": ["" for _ in inputs.oc_start_times],
            "stop": ["" for _ in inputs.oc_start_times],
            "RA": [0.0 for _ in inputs.oc_start_times],
            "DEC": [0.0 for _ in inputs.oc_start_times],
        }
    )

    starts_mjd = Time(list(inputs.oc_start_times), format="datetime", scale="utc").to_value("mjd")
    stops_mjd = Time(list(inputs.oc_stop_times), format="datetime", scale="utc").to_value("mjd")

    result_df, flag = observation_utils.schedule_occultation_targets(
        occ_list["Star Name"].to_numpy(),
        starts_mjd,
        stops_mjd,
        inputs.visit_start,
        inputs.visit_stop,
        data_dir / "aux_targets",
        occ_df.copy(),
        occ_list,
        "occ list",
    )
    return flag, result_df


def run_legacy_helper(module, inputs: OccultationInputs, data_dir: Path):
    occ_list = pd.read_csv(data_dir / "occultation-standard_targets.csv")
    occ_df = pd.DataFrame(
        {
            "Target": ["" for _ in inputs.oc_start_times],
            "start": ["" for _ in inputs.oc_start_times],
            "stop": ["" for _ in inputs.oc_start_times],
            "RA": [0.0 for _ in inputs.oc_start_times],
            "DEC": [0.0 for _ in inputs.oc_start_times],
        }
    )

    result_df, flag = module.sch_occ_new(  # type: ignore[attr-defined]
        list(inputs.oc_start_times),
        list(inputs.oc_stop_times),
        inputs.visit_start.to_pydatetime(),
        inputs.visit_stop.to_pydatetime(),
        data_dir / "occultation-standard_targets.csv",
        sort_key="closest",
        prev_obs=None,
    )
    return flag, result_df


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        legacy_module, legacy_root = load_legacy_module(args.legacy_module, tmp_path, args.schedule)

        legacy_data_dir = legacy_root / "data"

        data = gather_occultation_inputs(args.schedule, args.data_dir, args.row_index)

        print(
            f"Analysing row {data.schedule_index} ({data.target_name})"
            f" from {data.visit_start} to {data.visit_stop}"
        )
        print(f"Occultation windows: {len(data.oc_start_times)} intervals")

        legacy_flag, legacy_df = run_legacy_helper(legacy_module, data, legacy_data_dir)
        print(f"Legacy helper flag: {legacy_flag}")
        if legacy_df is not None:
            print(legacy_df.head())

        rework_flag, rework_df = run_rework_helper(data, args.data_dir)
        print(f"Rework helper flag: {rework_flag}")
        if rework_df is not None:
            print(rework_df.head())


if __name__ == "__main__":
    main()
