# Pandora Scheduler - Quick Start Guide

## Running the Complete Pipeline

## Running the Complete Pipeline

The `run_scheduler.py` script is the **main entry point** for the Pandora Scheduler. It runs the complete observation scheduling pipeline from start to finish, handling target manifest generation, visibility calculation, and schedule optimization.

### Prerequisites

The scheduler needs target definition files to generate target manifests and visibility data. You have two options:

1. **Use existing target manifests** (in `src/pandorascheduler/data/`)
2. **Generate from target definitions** (recommended for full pipeline)

#### Setting Up Target Definitions

The target definition files typically live in the `PandoraTargetList` repository. You can:

**Option A: Set environment variable**
```bash
export PANDORA_TARGET_DEFINITION_BASE=/Users/tsbarcl2/gitcode/PandoraTargetList/target_definition_files
```

**Option B: Specify on command line**
```bash
--target-definitions /Users/tsbarcl2/gitcode/PandoraTargetList/target_definition_files
```

### Basic Usage

```bash
poetry run python run_scheduler.py \
    --start "2026-02-05" \
    --end "2026-02-12" \
    --output ./output \
    --target-definitions /Users/tsbarcl2/gitcode/PandoraTargetList/target_definition_files \
    --generate-visibility
```

This will:
1. Generate target manifests from JSON definition files (exoplanet, auxiliary-standard, monitoring-standard, occultation-standard)
2. Generate visibility catalogs for all targets
3. Run the scheduling algorithm
4. Generate all output files

This will generate:
- **Target Manifests**: `exoplanet_targets.csv`, `auxiliary-standard_targets.csv`, etc. (in output/data/)
- **Visibility Files**: Time-series visibility data for each target (in output/data/targets/ and output/data/aux_targets/)
- **Schedule CSV**: Complete observation schedule
- **Science Calendar XML**: XML format for spacecraft planning
- **Observation Time Report**: Time allocation per target
- **Tracker files**: CSV and pickle formats for state tracking

### Output Files

When using `--target-definitions` and `--generate-visibility`, files are organized as:

```
output/
├── data/                                          # Generated data
│   ├── exoplanet_targets.csv                    # Primary targets manifest
│   ├── auxiliary-standard_targets.csv           # Auxiliary targets
│   ├── monitoring-standard_targets.csv          # Monitoring targets
│   ├── occultation-standard_targets.csv         # Occultation targets
│   ├── targets/                                  # Primary target visibility
│   │   └── StarName/
│   │       └── PlanetName/
│   │           └── Visibility for PlanetName.csv
│   └── aux_targets/                              # Auxiliary target visibility
│       └── StarName/
│           └── Visibility for StarName.csv
├── Pandora_Schedule_0.8_0.0_0.2_2026-02-05_to_2026-02-12.csv
├── Pandora_science_calendar.xml
├── Observation_Time_Report_2026-02-05 00:00:00.csv
├── tracker.csv
└── Tracker_2026-02-05_to_2026-02-12.pkl
```

When using existing manifests (no `--target-definitions`), only schedule/calendar files are created:

```
output/
├── Pandora_Schedule_0.8_0.0_0.2_2026-02-05_to_2026-02-12.csv
├── Pandora_science_calendar.xml
├── Observation_Time_Report_2026-02-05 00:00:00.csv
├── tracker.csv
└── Tracker_2026-02-05_to_2026-02-12.pkl
```

### Common Workflows

#### Complete Pipeline (Recommended)
Generate everything from scratch using target definitions:
```bash
poetry run python run_scheduler.py \
    --start "2026-02-05" \
    --end "2026-02-12" \
    --output ./full_pipeline_output \
    --target-definitions /Users/tsbarcl2/gitcode/PandoraTargetList/target_definition_files \
    --generate-visibility \
    --show-progress
```

#### Using Environment Variable
```bash
export PANDORA_TARGET_DEFINITION_BASE=/Users/tsbarcl2/gitcode/PandoraTargetList/target_definition_files

poetry run python run_scheduler.py \
    --start "2026-02-05" \
    --end "2026-02-12" \
    --output ./output \
    --generate-visibility \
    --show-progress
```

#### Using Existing Manifests (Fast)
If you already have generated target manifests and visibility files:
```bash
poetry run python run_scheduler.py \
    --start "2026-02-05" \
    --end "2026-02-12" \
    --output ./output
```
This uses the existing files in `src/pandorascheduler/data/`.

#### Show Progress Bars
```bash
poetry run python run_scheduler.py \
    --start "2026-02-05" \
    --end "2026-02-12" \
    --output ./output \
    --show-progress
```

#### Custom Weights
Schedule weights control prioritization (coverage, SAA, schedule factor):
```bash
poetry run python run_scheduler.py \
    --start "2026-02-05" \
    --end "2026-02-12" \
    --output ./output \
    --weights "0.9,0.0,0.1"
```

#### Skip XML Generation
For faster testing when you only need the CSV schedule:
```bash
poetry run python run_scheduler.py \
    --start "2026-02-05" \
    --end "2026-02-12" \
    --output ./output \
    --skip-xml
```

#### Verbose Logging
```bash
poetry run python run_scheduler.py \
    --start "2026-02-05" \
    --end "2026-02-12" \
    --output ./output \
    --verbose
```

### Advanced Usage

#### Using a Configuration File

Create `config.json`:
```json
{
    "obs_window_hours": 24.0,
    "transit_coverage_min": 0.4,
    "sched_weights": [0.8, 0.0, 0.2],
    "min_visibility": 0.5,
    "deprioritization_limit_hours": 48.0,
    "commissioning_days": 0,
    "aux_key": "sort_by_tdf_priority",
    "show_progress": true
}
```

Run with config:
```bash
poetry run python run_scheduler.py \
    --start "2026-02-05" \
    --end "2026-02-12" \
    --output ./output \
    --config config.json
```

A ready-made example config is provided at the repository root: `example_scheduler_config.json`.
For a full key reference see `docs/EXAMPLE_SCHEDULER_CONFIG.md`.


#### Visibility Generation

If you need to regenerate visibility catalogs from target definitions:
```bash
poetry run python run_scheduler.py \
    --start "2026-02-05" \
    --end "2026-02-12" \
    --output ./output \
    --target-definitions /path/to/PandoraTargetList/target_definition_files \
    --generate-visibility \
    --gmat-ephemeris /path/to/ephemeris.txt
```

**Note:** If `--target-definitions` is provided, the script will:
1. Generate `*_targets.csv` manifests in `output/data/`
2. If `--generate-visibility` is set, create visibility files in `output/data/targets/` and `output/data/aux_targets/`
3. Use these generated files for scheduling

#### Custom Avoidance Angles

```bash
poetry run python run_scheduler.py \
    --start "2026-02-05" \
    --end "2026-02-12" \
    --output ./output \
    --generate-visibility \
    --sun-avoidance 90.0 \
    --moon-avoidance 30.0 \
    --earth-avoidance 85.0
```

### Command-Line Reference

#### Required Arguments
- `--start DATE` - Schedule window start (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
- `--end DATE` - Schedule window end
- `--output DIR` - Output directory

#### Scheduling Parameters
- `--obs-window HOURS` - Observation window in hours (default: 24.0)
- `--transit-coverage FRACTION` - Minimum transit coverage (default: 0.4)
- `--weights W1,W2,W3` - Schedule weights: coverage,saa,schedule (default: 0.8,0.0,0.2)
- `--min-visibility FRACTION` - Minimum visibility for auxiliary targets (default: 0.5)

#### Visibility Generation
- `--generate-visibility` - Generate visibility catalogs
- `--gmat-ephemeris PATH` - GMAT ephemeris file
- `--sun-avoidance DEGREES` - Sun avoidance angle (default: 91.0)
- `--moon-avoidance DEGREES` - Moon avoidance angle (default: 25.0)
- `--earth-avoidance DEGREES` - Earth avoidance angle (default: 86.0)

#### Pipeline Control
- `--config PATH` - JSON configuration file
- `--target-definitions DIR` - Base directory for target definition files (PandoraTargetList/target_definition_files/)
- `--skip-xml` - Skip science calendar XML generation
- `--show-progress` - Show progress bars
- `-v, --verbose` - Enable verbose logging

**Environment Variable:**
- `PANDORA_TARGET_DEFINITION_BASE` - Alternative to `--target-definitions` flag

### Examples

#### Full Pipeline from Target Definitions
```bash
poetry run python run_scheduler.py \
    --start "2026-02-05" \
    --end "2026-02-12" \
    --output ./complete_run \
    --target-definitions /Users/tsbarcl2/gitcode/PandoraTargetList/target_definition_files \
    --generate-visibility \
    --show-progress \
    --verbose
```

#### Quick 2-Day Test with Existing Data
```bash
poetry run python run_scheduler.py \
    --start "2026-02-05" \
    --end "2026-02-07" \
    --output ./quick_test \
    --show-progress
```

#### Full 21-Day Run
```bash
poetry run python run_scheduler.py \
    --start "2026-02-01" \
    --end "2026-02-22" \
    --output ./full_run \
    --show-progress \
    --verbose
```

#### Custom Scheduling Strategy
```bash
poetry run python run_scheduler.py \
    --start "2026-02-05" \
    --end "2026-02-12" \
    --output ./custom \
    --weights "0.7,0.1,0.2" \
    --min-visibility 0.6 \
    --obs-window 20.0
```

### Troubleshooting

#### "Target manifest not found"
**Solution 1:** Provide target definitions to generate manifests:
```bash
--target-definitions /Users/tsbarcl2/gitcode/PandoraTargetList/target_definition_files
```

**Solution 2:** Set environment variable:
```bash
export PANDORA_TARGET_DEFINITION_BASE=/Users/tsbarcl2/gitcode/PandoraTargetList/target_definition_files
```

**Solution 3:** Ensure legacy data directory exists with pre-generated manifests in `src/pandorascheduler/data/`

#### "Target definition directory not found"
Check that the path to `PandoraTargetList/target_definition_files` is correct. It should contain subdirectories like:
- `exoplanet/`
- `auxiliary-standard/`
- `monitoring-standard/`
- `occultation-standard/`

Each containing JSON target definition files.

#### Visibility Generation Issues
If visibility generation fails:
1. Ensure `--target-definitions` is provided (manifests must exist first)
2. Check that GMAT ephemeris file exists (uses default if not specified)
3. Verify target manifest was generated successfully
4. Use `--verbose` to see detailed error messages

#### "Unable to locate target definition files"
The script searches for target definitions in this order:
1. `--target-definitions` command-line argument
2. `PANDORA_TARGET_DEFINITION_BASE` environment variable
3. Fallback to `comparison_outputs/target_definition_files_limited/` (limited test set)

For full runs, use option 1 or 2 pointing to the complete `PandoraTargetList` repository.

#### "Weights must sum to 1.0"
The three schedule weights must sum to exactly 1.0:
```bash
--weights "0.8,0.0,0.2"  # Valid: 0.8 + 0.0 + 0.2 = 1.0
--weights "0.9,0.0,0.2"  # Invalid: sum = 1.1
```

#### Memory Usage
For long scheduling windows (>30 days), the process may require significant memory. Consider:
- Running in smaller chunks
- Using `--skip-xml` to reduce memory usage
- Closing other applications

### Performance

Typical execution times on a modern laptop:
- 2-day window: ~2-3 minutes
- 7-day window: ~8-10 minutes
- 21-day window: ~25-30 minutes

The `--show-progress` flag provides real-time progress updates.

### Integration with Testing

To verify parity with legacy code:
```bash
# Run the main script
poetry run python run_scheduler.py \
    --start "2026-02-05" \
    --end "2026-02-12" \
    --output ./test_output

# Compare with legacy using comparison script
poetry run python scripts/run_schedule_comparison.py \
    --window-days 7
```
