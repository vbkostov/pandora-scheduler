# Target Definition Integration - Update Summary

## Changes Made

Updated `run_scheduler.py` to support complete end-to-end pipeline execution from target definition files, enabling generation of target manifests and visibility catalogs.

## Key Features Added

### 1. Target Definition File Support

The script now accepts a base directory containing target definition files (JSON format):

```bash
--target-definitions /Users/tsbarcl2/gitcode/PandoraTargetList/target_definition_files
```

Or via environment variable:
```bash
export PANDORA_TARGET_DEFINITION_BASE=/path/to/target_definition_files
```

### 2. Automatic Manifest Generation

When `--target-definitions` is provided, the script:
- Generates `*_targets.csv` manifests in `output/data/`
- Creates manifests for all 4 categories:
  - `exoplanet_targets.csv` (primary targets)
  - `auxiliary-standard_targets.csv`
  - `monitoring-standard_targets.csv`
  - `occultation-standard_targets.csv`

### 3. Visibility Catalog Generation

With both `--target-definitions` and `--generate-visibility`:
- Generates visibility time series for all targets
- Stores in `output/data/targets/` (primary) and `output/data/aux_targets/` (auxiliary)
- Uses GMAT ephemeris for orbit calculations
- Applies configurable avoidance angles (sun, moon, earth)

### 4. Three Usage Modes

#### Mode 1: Full Pipeline (Generate Everything)
```bash
poetry run python run_scheduler.py \
    --start "2026-02-05" \
    --end "2026-02-12" \
    --output ./output \
    --target-definitions /path/to/target_definition_files \
    --generate-visibility \
    --show-progress
```
Generates: Manifests → Visibility → Schedule → XML

#### Mode 2: Generate Manifests Only
```bash
poetry run python run_scheduler.py \
    --start "2026-02-05" \
    --end "2026-02-12" \
    --output ./output \
    --target-definitions /path/to/target_definition_files
```
Generates: Manifests → Schedule → XML (uses existing or on-demand visibility)

#### Mode 3: Use Existing Manifests (Original Behavior)
```bash
poetry run python run_scheduler.py \
    --start "2026-02-05" \
    --end "2026-02-12" \
    --output ./output
```
Uses: Existing manifests from `src/pandorascheduler/data/`

## Technical Implementation

### Environment Variable Resolution

The script checks for target definitions in this order:
1. `--target-definitions` command-line argument
2. `PANDORA_TARGET_DEFINITION_BASE` environment variable
3. If neither provided, uses existing manifests (no generation)

### Configuration Flow

```python
# In build_config_dict()
if args.target_definitions or os.environ.get("PANDORA_TARGET_DEFINITION_BASE"):
    config["target_definition_base"] = target_def_base
    logger.info("Will generate target manifests from: {path}")

if args.generate_visibility:
    config["generate_visibility"] = True
    # Requires target_definition_base to be set
```

### File Path Overrides

When using custom target definitions:
```python
extra_inputs = {
    "primary_target_csv": output_dir / "data" / "exoplanet_targets.csv",
    "auxiliary_target_csv": output_dir / "data" / "auxiliary-standard_targets.csv",
    "monitoring_target_csv": output_dir / "data" / "monitoring-standard_targets.csv",
    "occultation_target_csv": output_dir / "data" / "occultation-standard_targets.csv",
}
```

This tells the pipeline to use generated manifests instead of legacy data directory.

## Pipeline Integration

The script leverages existing `pandorascheduler_rework.pipeline` functionality:

1. **`_generate_target_manifests()`** - Called when `target_definition_base` is set
2. **`_maybe_generate_visibility()`** - Called when `generate_visibility` is True
3. Both functions are part of `build_schedule()` workflow

## Documentation Updates

### Updated Files

1. **`QUICK_START.md`**
   - Added prerequisites section explaining target definitions
   - Added three workflow examples (full pipeline, env var, existing manifests)
   - Expanded troubleshooting section
   - Added output structure documentation

2. **`EXAMPLE_FULL_PIPELINE.md`** (new)
   - Step-by-step example of complete pipeline
   - Shows environment variable setup
   - Documents expected output structure
   - Includes performance notes

## Testing

The changes maintain backward compatibility:

### Existing Behavior (No Changes)
```bash
poetry run python run_scheduler.py \
    --start "2026-02-05" --end "2026-02-12" \
    --output ./output
```
Still works with existing manifests in `src/pandorascheduler/data/`

### New Capability
```bash
export PANDORA_TARGET_DEFINITION_BASE=/Users/tsbarcl2/gitcode/PandoraTargetList/target_definition_files

poetry run python run_scheduler.py \
    --start "2026-02-05" --end "2026-02-12" \
    --output ./output \
    --generate-visibility \
    --show-progress
```
Generates everything from definitions

## Error Handling

### Improved Error Messages

**Before:**
```
Target manifest not found: /path/to/exoplanet_targets.csv
Please ensure the legacy data directory exists.
```

**After:**
```
Target manifest not found: /path/to/exoplanet_targets.csv
Please provide target definitions via --target-definitions or 
set PANDORA_TARGET_DEFINITION_BASE environment variable, 
or ensure the legacy data directory exists.
```

### Validation

- Checks if target definition directory exists before processing
- Warns if visibility generation requested without target definitions
- Only validates manifest existence when NOT generating from definitions

## Benefits

1. **Self-Contained Execution** - No dependency on pre-generated manifests
2. **Version Control** - Target definitions in JSON are easier to track than CSV
3. **Reproducibility** - Same definitions always produce same manifests
4. **Flexibility** - Can run with or without visibility generation
5. **Testing** - Easy to test with different target definition sets

## Future Enhancements

Potential improvements for future versions:

1. **Caching** - Reuse visibility files if they exist and are up-to-date
2. **Parallel Generation** - Generate visibility for multiple targets concurrently
3. **Incremental Updates** - Only regenerate changed targets
4. **Validation** - Verify target definition integrity before processing
5. **Progress Tracking** - Show detailed progress for visibility generation phase

## Migration Guide

### For Existing Users

No changes required! The script maintains full backward compatibility.

### For New Users (Recommended Workflow)

1. Clone PandoraTargetList repository
2. Set environment variable in your shell profile:
   ```bash
   export PANDORA_TARGET_DEFINITION_BASE=/path/to/PandoraTargetList/target_definition_files
   ```
3. Run pipeline with `--generate-visibility` on first run
4. Subsequent runs can omit `--generate-visibility` to reuse data

### For CI/CD

```bash
# Set in CI environment
export PANDORA_TARGET_DEFINITION_BASE=/workspace/PandoraTargetList/target_definition_files

# Run full pipeline
poetry run python run_scheduler.py \
    --start "${START_DATE}" \
    --end "${END_DATE}" \
    --output ./pipeline_output \
    --generate-visibility \
    --show-progress
```

## Summary

The updated `run_scheduler.py` script now provides a complete, self-contained pipeline that can:
- Generate target manifests from JSON definitions
- Create visibility catalogs from orbital ephemeris
- Run scheduling algorithms
- Produce all final outputs (schedule, XML, reports)

All with a single command, making it ideal for:
- Development and testing
- Automated pipelines
- Reproducible research
- Version-controlled target management
