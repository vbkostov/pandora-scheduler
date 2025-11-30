# Example: Running Full Pipeline with Target Definitions

This example demonstrates running the complete pipeline using `run_scheduler.py`, which is the recommended way to execute the scheduler.

## Setup

```bash
# Set the target definition base (one-time setup)
export PANDORA_TARGET_DEFINITION_BASE=/Users/tsbarcl2/gitcode/PandoraTargetList/target_definition_files
```

## Run Complete Pipeline

Generate everything from scratch:

```bash
poetry run python run_scheduler.py \
    --start "2026-02-05" \
    --end "2026-02-12" \
    --output ./complete_pipeline_run \
    --generate-visibility \
    --show-progress \
    --verbose
```

This will:
1. Read target definitions from `$PANDORA_TARGET_DEFINITION_BASE`
2. Generate `*_targets.csv` manifests in `complete_pipeline_run/data/`
3. Generate visibility files for all targets
4. Run the scheduling algorithm
5. Generate schedule CSV and science calendar XML

## Alternative: Specify Path Directly

```bash
poetry run python run_scheduler.py \
    --start "2026-02-05" \
    --end "2026-02-12" \
    --output ./complete_pipeline_run \
    --target-definitions /Users/tsbarcl2/gitcode/PandoraTargetList/target_definition_files \
    --generate-visibility \
    --show-progress
```

## Output Structure

```
complete_pipeline_run/
├── data/                                         # Generated from definitions
│   ├── exoplanet_targets.csv                   # Primary targets
│   ├── auxiliary-standard_targets.csv          # Auxiliary targets
│   ├── monitoring-standard_targets.csv         # Monitoring targets
│   ├── occultation-standard_targets.csv        # Occultation targets
│   ├── targets/                                 # Primary visibility data
│   │   └── [StarName]/
│   │       └── [PlanetName]/
│   │           └── Visibility for [PlanetName].csv
│   └── aux_targets/                             # Auxiliary visibility data
│       └── [StarName]/
│           └── Visibility for [StarName].csv
├── Pandora_Schedule_0.8_0.0_0.2_2026-02-05_to_2026-02-12.csv
├── Pandora_science_calendar.xml
├── Observation_Time_Report_2026-02-05 00:00:00.csv
├── tracker.csv
└── Tracker_2026-02-05_to_2026-02-12.pkl
```

## Notes

- The first run with `--generate-visibility` will take longer (visibility generation is compute-intensive)
- Subsequent runs can reuse the generated visibility files by omitting `--generate-visibility`
- The `--show-progress` flag shows progress bars for scheduling
- Use `--verbose` to see detailed logging including visibility generation status
