from __future__ import annotations

import shutil
import argparse
from datetime import datetime, timedelta
import importlib.util
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
from pandorascheduler_rework.pipeline import (
    SchedulerRequest,
    SchedulerResult,
    build_schedule,
)
from pandorascheduler_rework.utils.calendar_diff import compare_with_legacy


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run rework vs legacy scheduler comparison"
    )
    parser.add_argument(
        "--target-definition-base",
        dest="target_definition_base",
        help=(
            "Path to the PandoraTargetList target_definition_files directory to"
            " build manifests from before scheduling. Defaults to the value of"
            " PANDORA_TARGET_DEFINITION_BASE or the bundled fixtures if"
            " unspecified."
        ),
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=365,
        help="Number of days in the scheduling window (default: 365).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.window_days < 1:
        raise ValueError("--window-days must be at least 1")

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "src" / "pandorascheduler" / "data"
    comparison_dir = repo_root / "comparison_outputs"
    rework_dir = comparison_dir / "rework"
    legacy_dir = comparison_dir / "legacy"

    for path in (rework_dir, legacy_dir):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    base_start = datetime.fromisoformat("2026-02-05 00:00:00")
    window_start = base_start
    window_end = base_start + timedelta(days=args.window_days)
    pandora_start_str = window_start.strftime("%Y-%m-%d %H:%M:%S")
    pandora_stop_str = window_end.strftime("%Y-%m-%d %H:%M:%S")

    legacy_module = load_legacy_scheduler(repo_root)

    target_definition_files = [
        "exoplanet",
        "auxiliary-standard",
        "monitoring-standard",
        "occultation-standard",
    ]
    legacy_module.target_definition_files = target_definition_files  # type: ignore[attr-defined]

    config = {
        "pandora_start": pandora_start_str,
        "pandora_stop": pandora_stop_str,
        "obs_window_hours": 24.0,
        "transit_coverage_min": 0.4,
        "sched_weights": (0.8, 0.0, 0.2),
        "min_visibility": 0.5,
        "deprioritization_limit_hours": 48.0,
        "commissioning_days": 0,
        "aux_key": "sort_by_tdf_priority",
        "show_progress": True,
    }
    if args.target_definition_base:
        config["target_definition_base"] = args.target_definition_base

    request = SchedulerRequest(
        targets_manifest=data_dir / "baseline" / "fingerprints.json",
        window_start=window_start,
        window_end=window_end,
        output_dir=rework_dir,
        config=config,
    )

    rework_result = build_schedule(request)

    schedule_filename = (
        f"Pandora_Schedule_0.8_0.0_0.2_"
        f"{window_start:%Y-%m-%d}_to_{window_end:%Y-%m-%d}.csv"
    )
    if rework_result.schedule_csv is not None:
        schedule_filename = Path(rework_result.schedule_csv).name
    tracker_pickle_name = (
        f"Tracker_{window_start:%Y-%m-%d}_to_{window_end:%Y-%m-%d}.pkl"
    )

    legacy_module.Schedule(
        pandora_start_str,
        pandora_stop_str,
        str(data_dir / "exoplanet_targets.csv"),
        timedelta(hours=24.0),
        0.4,
        [0.8, 0.0, 0.2],
        0.5,
        48.0,
        aux_key="sort_by_tdf_priority",
    aux_list=str(data_dir / "occultation-standard_targets.csv"),
        fname_tracker=str(legacy_dir / tracker_pickle_name),
        commissioning_time=0,
        sched_start=pandora_start_str,
        sched_stop=pandora_stop_str,
        output_dir=str(legacy_dir),
    )

    comparisons = {
        "schedule": (
            rework_result.schedule_csv,
            legacy_dir / schedule_filename,
        ),
        "tracker_csv": (
            rework_result.reports.get("tracker_csv"),
            legacy_dir / "tracker.csv",
        ),
        "observation_report": (
            rework_result.reports.get("observation_time"),
            legacy_dir / f"Observation_Time_Report_{pandora_start_str}.csv",
        ),
    }

    tracker_pickle_paths = (
        rework_result.reports.get("tracker_pickle"),
        legacy_dir / tracker_pickle_name,
    )

    results = {
        key: compare_csv_pair(paths[0], paths[1]) for key, paths in comparisons.items()
    }

    if all(results.values()) and all(
        path.exists() for path in tracker_pickle_paths if path
    ):
        tracker_match = compare_tracker_pickles(tracker_pickle_paths)
    else:
        tracker_match = False

    results["tracker_pickle"] = tracker_match

    xml_match = False
    schedule_csv_path = rework_result.schedule_csv
    if schedule_csv_path is None:
        print("Rework schedule CSV missing; skipping XML comparison.")
    else:
        try:
            xml_result = compare_with_legacy(
                schedule_filename=schedule_filename,
                schedule_csv_path=schedule_csv_path,
            )
        except Exception as exc:  # pragma: no cover - surfaced via CLI output
            print(f"XML comparison failed: {exc}")
        else:
            xml_match = xml_result.match
            rework_xml_path = rework_dir / "Pandora_science_calendar.xml"
            legacy_xml_path = legacy_dir / "Pandora_science_calendar.xml"
            rework_xml_path.write_text(xml_result.rework_xml, encoding="utf-8")
            legacy_xml_path.write_text(xml_result.legacy_xml, encoding="utf-8")
            if not xml_result.match and xml_result.diff:
                diff_path = comparison_dir / "science_calendar.diff"
                diff_path.write_text(xml_result.diff + "\n", encoding="utf-8")
                print(f"XML diff written to {diff_path}")

    results["science_calendar_xml"] = xml_match

    print("Comparison summary:")
    for key, match in results.items():
        status = "MATCH" if match else "DIFF"
        print(f"  {key}: {status}")

    if not all(results.values()):
        print("Differing files:")
        for key, match in results.items():
            if not match:
                print(f"  - {key}")


def compare_csv_pair(rework_path: Path | None, legacy_path: Path) -> bool:
    if rework_path is None:
        return False
    if not rework_path.exists() or not legacy_path.exists():
        return False

    rework_df = pd.read_csv(rework_path)
    legacy_df = pd.read_csv(legacy_path)

    try:
        pd.testing.assert_frame_equal(rework_df, legacy_df, check_like=False)
    except AssertionError:
        report_difference(rework_df, legacy_df)
        return False
    return True


def report_difference(rework_df: pd.DataFrame, legacy_df: pd.DataFrame) -> None:
    joined = rework_df.merge(
        legacy_df,
        how="outer",
        indicator=True,
        suffixes=("_rework", "_legacy"),
    )
    mismatches = joined[joined["_merge"] != "both"]
    if not mismatches.empty:
        print("Rows present in only one output:")
        with pd.option_context("display.max_rows", 10, "display.width", 120):
            print(mismatches.head(10))
    else:
        deltas = (rework_df != legacy_df) & ~(rework_df.isna() & legacy_df.isna())
        differing_rows = deltas.any(axis=1)
        if differing_rows.any():
            print("Row-level differences detected:")
            with pd.option_context("display.max_rows", 10, "display.width", 200):
                print(rework_df[differing_rows].head(10))
                print(legacy_df[differing_rows].head(10))


def compare_tracker_pickles(paths: Iterable[Path | None]) -> bool:
    rework_path, legacy_path = paths
    if rework_path is None or legacy_path is None:
        return False
    if not rework_path.exists() or not legacy_path.exists():
        return False

    rework_df = pd.read_pickle(rework_path)
    legacy_df = pd.read_pickle(legacy_path)
    try:
        pd.testing.assert_frame_equal(rework_df, legacy_df, check_like=False)
    except AssertionError:
        print("Tracker pickle mismatch detected")
        return False
    return True


def load_legacy_scheduler(repo_root: Path):
    legacy_module_path = (
        repo_root / "src" / "pandorascheduler" / "scheduler_deprioritize_102925.py"
    )
    if not legacy_module_path.exists():
        raise FileNotFoundError(
            f"Legacy scheduler file not found: {legacy_module_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "pandorascheduler_legacy_scheduler", legacy_module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load legacy scheduler module spec")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module

    legacy_dir = str(legacy_module_path.parent)
    sys.path.insert(0, legacy_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(legacy_dir)

    return module


if __name__ == "__main__":
    main()
