# Pandora Scheduler Rework

A clean-room reimplementation of the Pandora scheduling pipeline lives in this
package.  The legacy code under `pandorascheduler` remains untouched so we can
continue producing trusted artifacts for regression testing.

## Goals

- Provide a clear, typed interface (`SchedulerRequest` → `SchedulerResult`).
- Consume explicit manifests for data inputs instead of hard-coded directories.
- Preserve parity with the existing outputs before opting into new features.
- Enable lightweight unit tests that validate the new pipeline incrementally.

## Next Steps

1. Activate the pre-existing mamba environment, then install dependencies with Poetry:

   ```bash
   conda activate <your-mamba-env>
   poetry install
   ```

2. Design adapters that read the current CSV/visibility assets into structured
   objects the new core can manipulate.
3. Implement the core scheduling logic with deterministic, testable
   components (Astropy remains the authoritative source for time/coordinate
   handling).
4. Cross-check results against the legacy pipeline using the fingerprint
   manifest in `src/pandorascheduler/data/baseline/fingerprints.json`.
