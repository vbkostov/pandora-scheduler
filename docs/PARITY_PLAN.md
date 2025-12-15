# Parity Plan: Validation & Acceptance

**Date:** 2025-11-22
**Status:** Active

## Objective
Ensure that the `pandorascheduler_rework` produces outputs that are functionally equivalent to the legacy system, with all differences being **intentional, understood, and documented**.

## 1. Current Parity Status

*   **Core Scheduling:** High parity. The scheduling algorithm (`scheduler.py`) produces identical schedules for the same inputs.
*   **Visibility:** High parity, with intentional improvements.
    *   *Difference:* No Simbad lookups (rework is offline-only).
    *   *Difference:* Corrected transit overlap calculation (legacy has a bug).
*   **Target Manifests:** High parity. Recent fixes (epoch, formatting) have aligned the outputs.
*   **Science Calendar:** High parity. XML structure matches, with minor differences in occultation window segmentation.

## 2. Validation Strategy

### A. Regression Testing
Use `scripts/run_schedule_comparison.py` to run side-by-side comparisons.
*   **Frequency:** Run on every significant logic change.
*   **Metrics:**
    *   Schedule CSV: Row-by-row comparison (Target, Start, Stop).
    *   Tracker CSV: Comparison of "Transits Left" and "Transit Priority".
    *   Visibility: Sampled comparison of generated visibility files.

### B. Acceptance Criteria
The rework is considered "ready" when:
1.  **Schedule Matches:** The primary target schedule matches the legacy schedule exactly (or differences are explained by the transit overlap fix).
2.  **No Crashes:** The rework runs end-to-end without errors for the full test dataset.
3.  **Data Integrity:** The rework correctly identifies and rejects bad input data (e.g., missing coordinates) where the legacy code might have silently failed or guessed.

## 3. Remaining Actions

### Short Term
- [x] **Verify Transit Overlap:** The "last partner wins" bug has been fixed in the legacy code.
- [ ] **Occultation Validation:** detailed comparison of `science_calendar.py` output vs `sched2xml_WIP.py` output for occultation targets.

### Long Term
- [ ] **Legacy Retirement:** Once the rework is validated, archive the `src/pandorascheduler` directory.
- [ ] **Data Migration:** Move `data/` to a top-level directory to decouple it from the legacy package structure.

## 4. Known "Won't Fix" Differences
These are intentional divergences where the rework is "correct" and the legacy is "wrong" or "outdated":
*   **Simbad Lookups:** We will NOT add Simbad lookups back to the rework.
*   **Transit Overlap Bug:** Fixed in legacy. Parity achieved.
