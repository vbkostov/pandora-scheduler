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

Example config.json:
{
    "obs_window_hours": 24.0,
    "transit_coverage_min": 0.4,
    "sched_weights": [0.8, 0.0, 0.2],
    "min_visibility": 0.5,
    "deprioritization_limit_hours": 48.0,
    "commissioning_days": 0,
    "aux_key": "sort_by_tdf_priority",
    "show_progress": true,
    "std_obs_duration_hours": 0.5,
    "std_obs_frequency_days": 3.0,
    "occ_sequence_limit_min": 50
}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from pandorascheduler_rework import SchedulerRequest, SchedulerResult, build_schedule
from pandorascheduler_rework import science_calendar


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


def load_config(config_path: Optional[Path]) -> Dict:
    """Load configuration from JSON file."""
    if config_path is None:
        return {}

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open() as f:
        return json.load(f)


def build_config_dict(args: argparse.Namespace, base_config: Dict, logger: logging.Logger) -> Dict:
    """Build final configuration dictionary from args and file config."""
    config = base_config.copy()

    # Override with command-line arguments
    config["obs_window_hours"] = args.obs_window
    config["transit_coverage_min"] = args.transit_coverage
    config["min_visibility"] = args.min_visibility
    config["show_progress"] = args.show_progress

    # Parse weights
    weights_str = args.weights.strip()
    weights = [float(w.strip()) for w in weights_str.split(",")]
    if len(weights) != 3:
        raise ValueError("Weights must have exactly 3 components")
    config["sched_weights"] = weights

    # Target definition base - check environment variable if not provided
    target_def_base = args.target_definitions
    if not target_def_base:
        import os
        env_value = os.environ.get("PANDORA_TARGET_DEFINITION_BASE")
        if env_value:
            target_def_base = Path(env_value).expanduser()
            logger.info(f"Using target definitions from PANDORA_TARGET_DEFINITION_BASE: {target_def_base}")
    
    if target_def_base:
        if not target_def_base.exists():
            raise FileNotFoundError(f"Target definition directory not found: {target_def_base}")
        config["target_definition_base"] = str(target_def_base.resolve())
        logger.info(f"Will generate target manifests from: {target_def_base}")

    # Visibility generation
    if args.generate_visibility:
        config["generate_visibility"] = True
        # # Set output_root for visibility if not provided in config
        # if "visibility_output_root" not in config:
        #     config["visibility_output_root"] = args.output / "data" / "targets"
        #     logger.info(f"Visibility output root set to default: {config['visibility_output_root']}")

        config["visibility_sun_deg"] = args.sun_avoidance
        config["visibility_moon_deg"] = args.moon_avoidance
        config["visibility_earth_deg"] = args.earth_avoidance

        if args.gmat_ephemeris:
            config["visibility_gmat"] = args.gmat_ephemeris
        
        if not target_def_base:
            logger.warning(
                "Visibility generation requested but no target definition base provided. "
                "Visibility will only be generated for existing manifests."
            )

    return config


def generate_science_calendar(
    schedule_csv: Path,
    output_dir: Path,
    data_dir: Path,
    logger: logging.Logger,
    config: Optional[object] = None,
) -> Path:
    """Generate science calendar XML from schedule CSV."""
    logger.info("Generating science calendar XML...")

    inputs = science_calendar.ScienceCalendarInputs(
        schedule_csv=schedule_csv,
        data_dir=data_dir,
    )

    output_path = output_dir / "Pandora_science_calendar.xml"

    science_calendar.generate_science_calendar(
        inputs,
        output_path=output_path,
        config=config,
    )

    logger.info(f"Science calendar written to: {output_path}")
    return output_path


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
        # Parse dates
        window_start = parse_datetime(args.start)
        window_end = parse_datetime(args.end)

        logger.info(f"Scheduling window: {window_start} to {window_end}")

        # Load configuration
        base_config = load_config(args.config)
        config = build_config_dict(args, base_config, logger)


        # Create output directory
        args.output.mkdir(parents=True, exist_ok=True)

        # Setup paths - rework ALWAYS generates data from target definitions
        output_data_dir = args.output / "data"
        output_data_dir.mkdir(parents=True, exist_ok=True)
        data_dir = output_data_dir
        
        # Target definitions are REQUIRED for the rework
        if "target_definition_base" not in config:
            logger.error("Target definition base is required for the rework scheduler")
            logger.error(
                "Please provide target definitions via --target-definitions or "
                "set PANDORA_TARGET_DEFINITION_BASE environment variable"
            )
            return 1
        
        logger.info("Generating target manifests from definition files...")
        
        # Define target categories
        target_categories = [
            "exoplanet",
            "auxiliary-standard", 
            "monitoring-standard",
            "occultation-standard",
        ]
        
        logger.info(f"Target manifests will be generated in: {data_dir}")
        
        # Determine primary target manifest path
        targets_manifest = data_dir / "exoplanet_targets.csv"

        # Build scheduler request - always use output data directory
        extra_inputs = {
            "primary_target_csv": data_dir / "exoplanet_targets.csv",
            "auxiliary_target_csv": data_dir / "auxiliary-standard_targets.csv",
            "monitoring_target_csv": data_dir / "monitoring-standard_targets.csv",
            "occultation_target_csv": data_dir / "occultation-standard_targets.csv",
        }
        
        request = SchedulerRequest(
            targets_manifest=targets_manifest,
            window_start=window_start,
            window_end=window_end,
            output_dir=args.output,
            config=config,
            extra_inputs=extra_inputs,
        )

        # Run scheduler
        logger.info("Starting scheduler pipeline...")
        result = build_schedule(request)

        # Generate science calendar XML (unless skipped)
        xml_path = None
        if not args.skip_xml and result.schedule_csv:
            # Use output data directory if it exists, otherwise fallback to source
            sc_data_dir = args.output / "data"
            if not sc_data_dir.exists():
                sc_data_dir = data_dir

            # Create config for science calendar generation
            from pandorascheduler_rework.science_calendar import ScienceCalendarConfig
            sc_config = ScienceCalendarConfig(
                visit_limit=None,
                prioritise_occultations_by_slew=False,
                occ_sequence_limit_min=config.get("occ_sequence_limit_min", 50),
            )

            xml_path = generate_science_calendar(
                result.schedule_csv,
                args.output,
                sc_data_dir,
                logger,
                config=sc_config,
            )

        # Print summary
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
