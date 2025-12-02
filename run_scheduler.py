#!/usr/bin/env python3
"""
Pandora Scheduler - Main Entry Point

This script runs the complete Pandora observation scheduling pipeline from start
to finish, generating all necessary output files including:
  - Target manifests (from target definition files)
  - Visibility catalogs (if requested)
  - Observation schedule CSV
  - Science calendar XML
  - Observation time reports
  - Tracker files (CSV and pickle)

Usage:
    # Basic run with default configuration
    poetry run python run_scheduler.py \\
        --start "2026-02-05" \\
        --end "2026-02-12" \\
        --output ./output

    # Run with custom configuration
    poetry run python run_scheduler.py \\
        --start "2026-02-05" \\
        --end "2026-02-12" \\
        --output ./output \\
        --config config.json

    # Generate visibility data as part of the run
    poetry run python run_scheduler.py \\
        --start "2026-02-05" \\
        --end "2026-02-12" \\
        --output ./output \\
        --generate-visibility

    # Use custom target definition files
    poetry run python run_scheduler.py \\
        --start "2026-02-05" \\
        --end "2026-02-12" \\
        --output ./output \\
        --target-definitions ./custom_targets
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any

from pandorascheduler_rework.config import PandoraSchedulerConfig
from pandorascheduler_rework.pipeline import build_schedule, SchedulerResult
from pandorascheduler_rework.science_calendar import (
    generate_science_calendar,
    ScienceCalendarInputs,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Pandora observation scheduler pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Schedule window start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="Schedule window end date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for generated files",
    )

    # Optional configuration
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to JSON configuration file (overrides defaults)",
    )
    parser.add_argument(
        "--target-definitions",
        type=Path,
        help=(
            "Base directory containing target definition files (e.g., PandoraTargetList/target_definition_files/). "
            "When provided, generates *_targets.csv manifests from JSON definitions. "
            "Can also be set via PANDORA_TARGET_DEFINITION_BASE environment variable."
        ),
    )
    parser.add_argument(
        "--generate-visibility",
        action="store_true",
        help="Generate visibility catalogs before scheduling (requires target manifests)",
    )

    # Visibility configuration (if generating)
    parser.add_argument(
        "--gmat-ephemeris",
        type=Path,
        help="Path to GMAT ephemeris file (for visibility generation)",
    )
    parser.add_argument(
        "--sun-avoidance",
        type=float,
        default=91.0,
        help="Sun avoidance angle in degrees (default: 91.0)",
    )
    parser.add_argument(
        "--moon-avoidance",
        type=float,
        default=25.0,
        help="Moon avoidance angle in degrees (default: 25.0)",
    )
    parser.add_argument(
        "--earth-avoidance",
        type=float,
        default=86.0,
        help="Earth avoidance angle in degrees (default: 86.0)",
    )

    # Scheduling configuration
    parser.add_argument(
        "--obs-window",
        type=float,
        default=24.0,
        help="Observation window in hours (default: 24.0)",
    )
    parser.add_argument(
        "--transit-coverage",
        type=float,
        default=0.4,
        help="Minimum transit coverage fraction (default: 0.4)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="0.8,0.0,0.2",
        help="Schedule weights as comma-separated values: coverage,saa,schedule (default: 0.8,0.0,0.2)",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.5,
        help="Minimum visibility fraction for auxiliary targets (default: 0.5)",
    )

    # Pipeline control
    parser.add_argument(
        "--skip-xml",
        action="store_true",
        help="Skip science calendar XML generation",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show progress bars during scheduling",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def parse_datetime(value: str) -> datetime:
    """Parse datetime string in flexible formats."""
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unable to parse datetime: {value}")


def load_json_config(config_path: Optional[Path]) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    if config_path is None:
        return {}

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open() as f:
        return json.load(f)


def create_scheduler_config(
    args: argparse.Namespace, 
    json_config: Dict[str, Any], 
    logger: logging.Logger
) -> PandoraSchedulerConfig:
    """Create unified configuration object from args and file config."""
    
    # 1. Parse weights
    weights_str = args.weights.strip()
    weights_list = [float(w.strip()) for w in weights_str.split(",")]
    if len(weights_list) != 3:
        raise ValueError("Weights must have exactly 3 components")
    sched_weights = (weights_list[0], weights_list[1], weights_list[2])

    # 2. Determine target definition base
    target_def_base = args.target_definitions
    if not target_def_base:
        env_value = os.environ.get("PANDORA_TARGET_DEFINITION_BASE")
        if env_value:
            target_def_base = Path(env_value).expanduser()
            logger.info(f"Using target definitions from PANDORA_TARGET_DEFINITION_BASE: {target_def_base}")
    
    # 3. Determine visibility settings
    visibility_gmat = args.gmat_ephemeris
    if not visibility_gmat and "visibility_gmat" in json_config:
        visibility_gmat = Path(json_config["visibility_gmat"])

    # 4. Construct Config Object
    # Priority: Args > JSON > Defaults (handled by dataclass)
    
    # Helper to get value from JSON or Args or Default
    def get_val(key: str, arg_val: Any, default: Any = None) -> Any:
        if arg_val != default:
            return arg_val
        return json_config.get(key, arg_val)

    # Build PandoraSchedulerConfig with the dataclass field names and types
    obs_window_hours = float(get_val("obs_window_hours", args.obs_window, 24.0))
    transit_cov = float(get_val("transit_coverage_min", args.transit_coverage, 0.4))
    min_vis = float(get_val("min_visibility", args.min_visibility, 0.5))

    # Coerce unified transit_scheduling_weights from JSON or CLI into a 3-tuple
    raw_transit_weights = (
        json_config.get("transit_scheduling_weights")
        or sched_weights
    )
    if isinstance(raw_transit_weights, str):
        raw_transit_weights = tuple(float(x.strip()) for x in raw_transit_weights.split(","))
    transit_weights_tuple = tuple(float(x) for x in raw_transit_weights)

    extra_inputs: Dict[str, Any] = {}
    if target_def_base:
        extra_inputs["target_definition_base"] = Path(target_def_base)
        # When target definitions are provided, we need to specify which categories to process.
        # These map to the standard directory names in the PandoraTargetList repository.
        extra_inputs["target_definition_files"] = [
            "exoplanet",
            "auxiliary-standard",
            "monitoring-standard",
            "occultation-standard",
        ]

    # Visibility GMAT goes into the typed field `gmat_ephemeris` on the config
    if visibility_gmat is None:
        gmat_path = None
    else:
        gmat_path = Path(str(visibility_gmat)).expanduser().resolve()

    config = PandoraSchedulerConfig(
        window_start=parse_datetime(args.start),
        window_end=parse_datetime(args.end),
        obs_window=timedelta(hours=obs_window_hours),
        targets_manifest=args.output / "data",
        gmat_ephemeris=gmat_path,
        output_dir=args.output,
        
        # Scheduling Thresholds
        transit_coverage_min=transit_cov,
        min_visibility=min_vis,
        deprioritization_limit_hours=float(json_config.get("deprioritization_limit_hours", 48.0)),
        saa_overlap_threshold=float(json_config.get("saa_overlap_threshold", 0.0)),
        commissioning_days=int(json_config.get("commissioning_days", 0)),
        
        # Weights
        transit_scheduling_weights=transit_weights_tuple,
        
        # Keepout Angles
        sun_avoidance_deg=float(get_val("visibility_sun_deg", args.sun_avoidance, 91.0)),
        moon_avoidance_deg=float(get_val("visibility_moon_deg", args.moon_avoidance, 25.0)),
        earth_avoidance_deg=float(get_val("visibility_earth_deg", args.earth_avoidance, 86.0)),
        
        # XML Generation Parameters
        obs_sequence_duration_min=int(json_config.get("obs_sequence_duration_min", 90)),
        occ_sequence_limit_min=int(json_config.get("occ_sequence_limit_min", 50)),
        min_sequence_minutes=int(json_config.get("min_sequence_minutes", 5)),
        break_occultation_sequences=bool(json_config.get("break_occultation_sequences", True)),
        
        # Standard Observations
        std_obs_duration_hours=float(json_config.get("std_obs_duration_hours", 0.5)),
        std_obs_frequency_days=float(json_config.get("std_obs_frequency_days", 3.0)),
        
        # Behavior Flags
        show_progress=bool(get_val("show_progress", args.show_progress, False)),
        force_regenerate=bool(json_config.get("force_regenerate", False)),
        use_target_list_for_occultations=bool(json_config.get("use_target_list_for_occultations", False)),
        prioritise_occultations_by_slew=bool(json_config.get("prioritise_occultations_by_slew", False)),
        
        # Auxiliary Sorting
        aux_sort_key=str(json_config.get("aux_sort_key", "sort_by_tdf_priority")),
        
        # Metadata
        author=json_config.get("author"),
        created_timestamp=json_config.get("created_timestamp"),
        visit_limit=json_config.get("visit_limit"),  # None by default
        target_filters=json_config.get("target_filters", []),
        
        extra_inputs={**json_config.get("extra_inputs", {}), **extra_inputs},
    )
    
    return config


def print_summary(result: SchedulerResult, xml_path: Optional[Path]) -> None:
    """Print summary of generated outputs."""
    print("\n" + "=" * 80)
    print("SCHEDULING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print("\nGenerated Files:")
    print("-" * 80)

    if result.schedule_csv:
        print(f"  📄 Schedule CSV:      {result.schedule_csv}")

    for name, path in result.reports.items():
        label = name.replace("_", " ").title()
        print(f"  📊 {label:18} {path}")

    if xml_path:
        print(f"  📑 Science Calendar:  {xml_path}")

    print("-" * 80)

    # Print statistics
    if result.diagnostics:
        schedule_df = result.diagnostics.get("schedule_dataframe")
        if schedule_df is not None and not schedule_df.empty:
            print(f"\nSchedule Statistics:")
            print(f"  Total observations:   {len(schedule_df)}")
            print(f"  Unique targets:       {schedule_df['Target'].nunique()}")

            # Count primary vs auxiliary
            primary_count = schedule_df[
                ~schedule_df["Target"].str.contains("STD|Free Time", na=False)
            ].shape[0]
            print(f"  Primary observations: {primary_count}")

    print("\n" + "=" * 80 + "\n")


def main() -> int:
    """Main entry point for the scheduler pipeline."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Scheduling window: {args.start} to {args.end}")

        # 1. Load and Build Configuration
        json_config = load_json_config(args.config)
        config = create_scheduler_config(args, json_config, logger)

        # 2. Validate Requirements
        # The unified config stores optional extra inputs; prefer CLI arg, then env var,
        # then any value passed via config.extra_inputs (where create_scheduler_config
        # may have placed the target_definition_base).
        target_def_present = bool(
            args.target_definitions
            or os.environ.get("PANDORA_TARGET_DEFINITION_BASE")
            or (hasattr(config, "extra_inputs") and config.extra_inputs.get("target_definition_base"))
        )
        if not target_def_present:
            logger.error("Target definition base is required for the rework scheduler")
            logger.error(
                "Please provide target definitions via --target-definitions or "
                "set PANDORA_TARGET_DEFINITION_BASE environment variable"
            )
            return 1
            
        # Determine whether visibility generation was requested (CLI or JSON)
        generate_visibility = args.generate_visibility or json_config.get("generate_visibility", False)
        if generate_visibility and not config.gmat_ephemeris:
            logger.warning("Visibility generation requested but no GMAT ephemeris provided.")

        # 3. Ensure targets manifest location exists (may be an output/data dir)
        if config.targets_manifest and not config.targets_manifest.exists():
            try:
                config.targets_manifest.mkdir(parents=True, exist_ok=True)
                logger.info("Created targets manifest directory: %s", config.targets_manifest)
            except Exception:
                logger.debug("Unable to create targets manifest directory: %s", config.targets_manifest)

        # 4. Run Scheduler (using new API)
        logger.info("Starting scheduler pipeline...")
        result = build_schedule(config)

        # 4. Generate Science Calendar XML
        xml_path = None
        if not args.skip_xml and result.schedule_csv:
            # Use the same config object to create calendar settings
            calendar_config = config.to_science_calendar_config()
            
            # Ensure we point to the correct data directory (where manifests are)
            # The pipeline copies/generates data into output_dir/data
            data_dir = config.output_dir / "data"
            
            inputs = ScienceCalendarInputs(
                schedule_csv=result.schedule_csv,
                data_dir=data_dir,
            )
            
            xml_path = generate_science_calendar(
                inputs=inputs,
                config=calendar_config,
            )
            logger.info(f"Science calendar written to: {xml_path}")

        # 5. Print Summary
        print_summary(result, xml_path)

        return 0

    except KeyboardInterrupt:
        logger.error("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
