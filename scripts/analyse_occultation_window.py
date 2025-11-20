#!/usr/bin/env python3
"""Inspect occultation segmentation for a specific visit.

This helper instantiates the rework ``_ScienceCalendarBuilder`` so we can
compare how visibility sampling and occultation selection behave for a chosen
row in the schedule CSV.  It prints the visibility cadence, the inferred
occultation windows, and the targets that ``schedule_occultation_targets``
assigns to each gap.  Passing ``--compare-data-dir`` runs the same analysis a
second time against an alternate data root (e.g. the legacy bundle) so we can
spot behavioural differences side by side.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import pandas as pd

from pandorascheduler_rework import observation_utils
from pandorascheduler_rework.xml_builder import (
    ScienceCalendarConfig,
    ScienceCalendarInputs,
    _ScienceCalendarBuilder,
    _extract_visibility_segment,
    _lookup_auxiliary_row,
    _lookup_planet_row,
    _normalise_target_name,
    _occultation_windows,
    _read_planet_visibility,
    _read_visibility,
    _resolve_coordinates,
    _transit_windows,
    _visibility_change_indices,
)


@dataclass(frozen=True)
class OccultationAnalysis:
    label: str
    visit_index: int
    target_label: str
    observation_start: pd.Timestamp
    observation_stop: pd.Timestamp
    visit_times: List[pd.Timestamp]
    visibility_flags: List[int]
    visibility_change_indices: List[int]
    occultation_starts: List[pd.Timestamp]
    occultation_stops: List[pd.Timestamp]
    occultation_targets: Optional[pd.DataFrame]


@contextlib.contextmanager
def override_data_roots(data_dir: Path) -> Iterator[None]:
    original = list(observation_utils.DATA_ROOTS)
    observation_utils.DATA_ROOTS = [data_dir]
    try:
        yield
    finally:
        observation_utils.DATA_ROOTS = original


def build_builder(schedule_csv: Path, data_dir: Path) -> _ScienceCalendarBuilder:
    inputs = ScienceCalendarInputs(schedule_csv=schedule_csv, data_dir=data_dir)
    config = ScienceCalendarConfig(visit_limit=None, show_progress=False)
    return _ScienceCalendarBuilder(inputs, config)


def analyse_occultations(
    *,
    label: str,
    builder: _ScienceCalendarBuilder,
    visit_index: int,
) -> OccultationAnalysis:
    schedule = builder.schedule
    if visit_index < 0 or visit_index >= len(schedule):
        raise IndexError(f"Visit index {visit_index} out of range (0..{len(schedule)-1})")

    row = schedule.iloc[visit_index]
    target_label = str(row.get("Target", ""))
    start = pd.to_datetime(row["Observation Start"])
    stop = pd.to_datetime(row["Observation Stop"])

    target_name, star_name = _normalise_target_name(target_label)
    has_transit = pd.notna(row.get("Transit Coverage"))

    target_info: Optional[pd.DataFrame]
    if has_transit:
        target_info = _lookup_planet_row(builder.target_catalog, target_name)
    else:
        target_info = _lookup_auxiliary_row(builder.aux_catalog, target_name)

    try:
        if target_info is not None and not target_info.empty:
            ra_value = float(target_info.iloc[0]["RA"])
            dec_value = float(target_info.iloc[0]["DEC"])
        else:
            ra_value, dec_value = _resolve_coordinates(star_name)
    except Exception:
        ra_value, dec_value = _resolve_coordinates(star_name)

    if has_transit:
        visibility_df = _read_visibility(builder.data_dir / "targets" / star_name, star_name)
        transit_df = _read_planet_visibility(
            builder.data_dir / "targets" / star_name / target_name,
            target_name,
        )
        transit_windows = _transit_windows(transit_df) if transit_df is not None else None
    else:
        visibility_df = _read_visibility(builder.data_dir / "aux_targets" / star_name, target_name)
        transit_windows = None

    if visibility_df is None or visibility_df.empty:
        raise FileNotFoundError(f"Visibility file missing for {star_name}/{target_name}")

    raw_visit_times, visibility_flags = _extract_visibility_segment(
        visibility_df,
        start.to_pydatetime(),
        stop.to_pydatetime(),
        builder.config.min_sequence_minutes,
    )

    visit_times = [pd.Timestamp(value) for value in raw_visit_times]

    change_indices = _visibility_change_indices(visibility_flags)
    raw_occ_starts, raw_occ_stops, _ = _occultation_windows(
        raw_visit_times,
        visibility_flags,
        change_indices,
    )

    occ_starts = [pd.Timestamp(value) for value in raw_occ_starts]
    occ_stops = [pd.Timestamp(value) for value in raw_occ_stops]

    occ_targets: Optional[pd.DataFrame] = None
    if occ_starts and occ_stops:
        result = builder._find_occultation_target(  # type: ignore[attr-defined]
            raw_occ_starts,
            raw_occ_stops,
            raw_visit_times[0],
            raw_visit_times[-1],
            ra_value,
            dec_value,
        )
        if result is not None:
            occ_targets, _ = result

    if transit_windows is not None:
        transit_windows = (
            [pd.Timestamp(value) for value in transit_windows[0]],
            [pd.Timestamp(value) for value in transit_windows[1]],
        )

    print_analysis(
        label=label,
        target_label=target_label,
        start=start,
        stop=stop,
        visit_times=visit_times,
        visibility_flags=visibility_flags,
        change_indices=change_indices,
        occ_starts=occ_starts,
        occ_stops=occ_stops,
        occ_targets=occ_targets,
        transit_windows=transit_windows,
    )

    return OccultationAnalysis(
        label=label,
        visit_index=visit_index,
        target_label=target_label,
        observation_start=start,
        observation_stop=stop,
        visit_times=visit_times,
        visibility_flags=visibility_flags,
        visibility_change_indices=change_indices,
        occultation_starts=occ_starts,
        occultation_stops=occ_stops,
        occultation_targets=occ_targets,
    )


def print_analysis(
    *,
    label: str,
    target_label: str,
    start: pd.Timestamp,
    stop: pd.Timestamp,
    visit_times: Sequence[pd.Timestamp],
    visibility_flags: Sequence[int],
    change_indices: Sequence[int],
    occ_starts: Sequence[pd.Timestamp],
    occ_stops: Sequence[pd.Timestamp],
    occ_targets: Optional[pd.DataFrame],
    transit_windows: Optional[tuple[Sequence[pd.Timestamp], Sequence[pd.Timestamp]]],
) -> None:
    print(f"\n=== {label} ===")
    print(f"Target: {target_label}")
    print(f"Visit:  {start} -> {stop} ({len(visit_times)} samples)")

    window_preview = list(zip(visit_times, visibility_flags))[:20]
    print("First 20 samples:")
    for timestamp, flag in window_preview:
        print(f"  {timestamp}  visible={flag}")

    print("Visibility change indices:", list(change_indices))

    if transit_windows is not None:
        t_start, t_stop = transit_windows
        print("Transit windows (start -> stop):")
        for ts, te in zip(t_start, t_stop):
            print(f"  {ts} -> {te}")

    if occ_starts:
        print("Occultation intervals:")
        for o_start, o_stop in zip(occ_starts, occ_stops):
            print(f"  {o_start} -> {o_stop} ({(o_stop - o_start).total_seconds()/60:.1f} min)")
    else:
        print("Occultation intervals: none")

    if occ_targets is not None:
        print("Occultation target assignments:")
        print(occ_targets.to_string(index=False))
    else:
        print("Occultation target assignments: none")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "schedule_csv",
        type=Path,
        help="Path to the schedule CSV to analyse",
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Root directory containing visibility and catalog CSVs",
    )
    parser.add_argument(
        "--visit-index",
        type=int,
        default=0,
        help="Zero-based row index in the schedule CSV (default: 0)",
    )
    parser.add_argument(
        "--compare-data-dir",
        type=Path,
        help="Optional second data directory to analyse for comparison",
    )
    parser.add_argument(
        "--compare-label",
        default="compare",
        help="Label for the comparison dataset (default: 'compare')",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    analyses: List[OccultationAnalysis] = []

    with override_data_roots(args.data_dir):
        builder = build_builder(args.schedule_csv, args.data_dir)
        analyses.append(
            analyse_occultations(label="primary", builder=builder, visit_index=args.visit_index)
        )

    if args.compare_data_dir is not None:
        with override_data_roots(args.compare_data_dir):
            compare_builder = build_builder(args.schedule_csv, args.compare_data_dir)
            analyses.append(
                analyse_occultations(
                    label=args.compare_label,
                    builder=compare_builder,
                    visit_index=args.visit_index,
                )
            )

    if len(analyses) == 2:
        compare_occultation_sets(analyses[0], analyses[1])


def compare_occultation_sets(
    first: OccultationAnalysis,
    second: OccultationAnalysis,
) -> None:
    print("\n=== comparison summary ===")
    print(f"Primary occultation count: {len(first.occultation_starts)}")
    print(f"Compare occultation count: {len(second.occultation_starts)}")

    primary_pairs = set(zip(first.occultation_starts, first.occultation_stops))
    compare_pairs = set(zip(second.occultation_starts, second.occultation_stops))

    missing = sorted(primary_pairs - compare_pairs)
    extra = sorted(compare_pairs - primary_pairs)

    if missing:
        print("Occultation intervals missing from comparison:")
        for start, stop in missing:
            print(f"  {start} -> {stop}")
    if extra:
        print("Occultation intervals only in comparison:")
        for start, stop in extra:
            print(f"  {start} -> {stop}")

    if first.occultation_targets is not None and second.occultation_targets is not None:
        primary_targets = tuple(first.occultation_targets.get("Target", []))
        compare_targets = tuple(second.occultation_targets.get("Target", []))
        if primary_targets != compare_targets:
            print("Occultation target order differs:")
            print("  primary:", primary_targets)
            print("  compare:", compare_targets)
        else:
            print("Occultation target order matches")


if __name__ == "__main__":
    main()
