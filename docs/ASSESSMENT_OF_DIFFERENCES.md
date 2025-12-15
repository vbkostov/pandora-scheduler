# Assessment of Output Differences: Legacy vs. Rework

**Date:** 2025-11-22
**Status:** Active Analysis

## Executive Summary

The `pandorascheduler_rework` aims for functional parity with the legacy `pandorascheduler` while modernizing the architecture. While the core scheduling logic produces identical results for the vast majority of targets, specific architectural decisions and bug fixes in the rework lead to intentional divergences.

## 1. Coordinate Resolution (High Divergence Potential)

*   **Legacy Behavior:** The legacy code (`helper_codes.py`) often attempts to resolve coordinates via SIMBAD (`SkyCoord.from_name`) if they are missing or malformed in the input CSVs.
*   **Rework Behavior:** The rework **strictly** relies on the input catalog/manifest values. It explicitly raises a `RuntimeError` if coordinates are missing.
*   **Impact:**
    *   **Reproducibility:** The rework is deterministic and works offline. The legacy code depends on network availability and the state of the SIMBAD database.
    *   **Data Integrity:** The rework forces users to fix their input data rather than silently masking missing coordinates.
    *   **Difference:** If the input catalog differs from SIMBAD, the calculated visibility windows and observation times will differ.

## 2. Visibility Calculation (Low Divergence)

*   **Legacy Behavior:**
    *   **Fixed:** The "last partner wins" bug in `transits.py` has been patched to correctly calculate maximum overlap.
    *   Skips visibility generation if files exist, even if parameters (like avoidance angles) have changed.
*   **Rework Behavior:**
    *   Correctly calculates the maximum overlap across all partners.
    *   Implements robust freshness checks.
*   **Impact:**
    *   **Transit Overlap:** Values should now match between legacy and rework for multi-planet systems.
    *   **Freshness:** The rework ensures visibility data matches the current configuration.

## 3. Science Calendar XML (Low Divergence)

*   **Legacy Behavior:** `sched2xml_WIP.py` has complex, ad-hoc logic for slicing occultation windows and handling edge cases.
*   **Rework Behavior:** `science_calendar.py` uses a simplified, cleaner logic for window segmentation.
*   **Impact:**
    *   **Occultations:** Start/stop times for occultation sequences might differ by small amounts (e.g., 1 minute) due to different rounding or segmentation logic.
    *   **Structure:** The XML structure is identical, but attribute ordering might vary (though XML is order-independent for attributes).

## 4. Target Manifest Generation (Low Divergence)

*   **Legacy Behavior:** Implicitly relies on directory structures and specific file naming conventions.
*   **Rework Behavior:** Uses `targets/manifest.py` to explicitly parse JSON definitions into standardized CSVs.
*   **Impact:**
    *   **Column Types:** The rework enforces stricter types (e.g., ensuring `Priority` is float), which might lead to minor differences in sorting if the legacy code relied on string comparison for numbers.
    *   **Parity:** Recent fixes (e.g., planet name formatting, epoch handling) have brought the rework into very close alignment with legacy.

## 5. Execution Model

*   **Legacy:** Script-based (`scheduler_deprioritize_102925.py`), relying on global state and hardcoded paths.
*   **Rework:** Library-based (`pipeline.py`, `scheduler.py`), using explicit configuration objects and dependency injection.
*   **Impact:** The rework is easier to test and integrate, but requires a different invocation method (`run_scheduler.py`).

## Conclusion

The primary sources of divergence are **intentional improvements**: removing network dependencies (Simbad) and fixing logic bugs (transit overlap). Any remaining differences should be validated to ensure they stem from these known causes.
