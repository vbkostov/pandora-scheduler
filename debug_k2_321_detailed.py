#!/usr/bin/env python3
"""Detailed debug of K2-321 occultation scheduling."""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from astropy.time import Time

# Add src to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root / "src"))

from pandorascheduler_rework import observation_utils, science_calendar

# Set up data directory
data_dir = repo_root / "src" / "pandorascheduler" / "data"
observation_utils.DATA_ROOTS = [data_dir]

# K2-321 visit parameters
target_name = "K2-321"
star_name = "K2-321"
visit_start = datetime(2026, 2, 10, 19, 25, 0)
visit_stop = datetime(2026, 2, 11, 17, 49, 0)

print("=" * 80)
print(f"Debugging occultation scheduling for {target_name}")
print(f"Visit: {visit_start} to {visit_stop}")
print("=" * 80)

# Load visibility data
vis_file = data_dir / "aux_targets" / star_name / f"Visibility for {target_name}.csv"
vis_df = pd.read_csv(vis_file)

# Extract visibility segment
vis_times_mjd = vis_df["Time(MJD_UTC)"].to_numpy()
vis_times_dt = Time(vis_times_mjd, format="mjd", scale="utc").to_datetime()
visibility_flags = vis_df["Visible"].to_numpy()

mask = (vis_times_dt >= visit_start) & (vis_times_dt <= visit_stop)
visit_times = vis_times_dt[mask]
visit_flags = visibility_flags[mask]

# Apply rework's visibility processing
def round_to_nearest_second(timestamp):
    if timestamp.microsecond >= 500_000:
        return timestamp + timedelta(seconds=1) - timedelta(microseconds=timestamp.microsecond)
    return timestamp - timedelta(microseconds=timestamp.microsecond)

visit_times = np.vectorize(round_to_nearest_second)(visit_times)

# Remove short sequences (5 minutes minimum)
min_seq_minutes = 5
if len(visit_flags) > 0:
    # Simple implementation
    visit_flags_filtered = visit_flags.copy()

print(f"\nProcessed visibility samples: {len(visit_times)}")
print(f"Visible: {np.sum(visit_flags == 1)}, Non-visible: {np.sum(visit_flags == 0)}")

# Find visibility changes
if len(visit_flags) > 1:
    changes = np.where(visit_flags[:-1] != visit_flags[1:])[0]
    print(f"Visibility changes: {len(changes)}")
    
    # Build occultation windows
    oc_starts = []
    oc_stops = []
    
    if not visit_flags[-1]:
        changes_list = changes.tolist()
        changes_list.append(len(visit_times) - 2)
        changes = np.array(changes_list)
    
    if not visit_flags[0]:
        oc_starts.append(visit_times[0])
        oc_stops.append(visit_times[changes[0]])
    
    for v in range(len(changes) - 1):
        if not visit_flags[changes[v] + 1]:
            oc_starts.append(visit_times[changes[v] + 1])
            oc_stops.append(visit_times[changes[v + 1]])
    
    print(f"\nOccultation windows before breaking: {len(oc_starts)}")
    for i, (start, stop) in enumerate(list(zip(oc_starts, oc_stops))[:5]):
        duration = (stop - start).total_seconds() / 60
        print(f"  {i+1}. {start} to {stop} ({duration:.1f} min)")
    
    # Break long sequences (31 minutes max)
    occultation_limit = timedelta(minutes=31)
    expanded_starts = []
    expanded_stops = []
    
    for start, stop in zip(oc_starts, oc_stops):
        segments = observation_utils.break_long_sequences(start, stop, occultation_limit)
        if not segments:
            expanded_starts.append(start)
            expanded_stops.append(stop)
        else:
            for seg_start, seg_stop in segments:
                expanded_starts.append(seg_start)
                expanded_stops.append(seg_stop)
    
    print(f"\nOccultation windows after breaking: {len(expanded_starts)}")
    for i, (start, stop) in enumerate(list(zip(expanded_starts, expanded_stops))[:10]):
        duration = (stop - start).total_seconds() / 60
        print(f"  {i+1}. {start} to {stop} ({duration:.1f} min)")
    
    # Now try to schedule occultation targets
    print("\n" + "=" * 80)
    print("Attempting to schedule occultation targets")
    print("=" * 80)
    
    # Get K2-321 coordinates for reference
    aux_catalog = pd.read_csv(data_dir / "aux_list_new.csv")
    k2_321_row = aux_catalog[aux_catalog["Star Name"] == "K2-321"]
    if not k2_321_row.empty:
        ref_ra = float(k2_321_row.iloc[0]["RA"])
        ref_dec = float(k2_321_row.iloc[0]["DEC"])
        print(f"\nReference coordinates (K2-321): RA={ref_ra}, DEC={ref_dec}")
    else:
        ref_ra, ref_dec = 0.0, 0.0
        print("\nWARNING: Could not find K2-321 in aux catalog")
    
    # Try occultation-standard list first
    occ_path = data_dir / "occultation-standard_targets.csv"
    vis_root = data_dir / "aux_targets"
    
    print(f"\nUsing occultation list: {occ_path}")
    print(f"Visibility root: {vis_root}")
    
    # Prepare schedule dataframe
    schedule_rows = [
        [
            "",
            start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            stop.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "",
            "",
        ]
        for start, stop in zip(expanded_starts, expanded_stops)
    ]
    occ_df = pd.DataFrame(schedule_rows, columns=["Target", "start", "stop", "RA", "DEC"])
    
    occ_list = pd.read_csv(occ_path)
    print(f"Occultation targets available: {len(occ_list)}")
    
    # Call the actual scheduling function
    target_names = occ_list["Star Name"].to_numpy()
    starts_mjd = Time(expanded_starts, format="datetime", scale="utc").to_value("mjd")
    stops_mjd = Time(expanded_stops, format="datetime", scale="utc").to_value("mjd")
    
    print(f"\nScheduling {len(expanded_starts)} occultation windows...")
    print(f"Trying {len(target_names)} potential targets...")
    
    result_df, flag = observation_utils.schedule_occultation_targets(
        target_names,
        starts_mjd,
        stops_mjd,
        visit_start,
        visit_stop,
        str(vis_root),
        occ_df,
        occ_list,
        "occ list",
    )
    
    print(f"\nScheduling successful: {flag}")
    if flag:
        filled = result_df[result_df["Target"].notna() & (result_df["Target"] != "")].shape[0]
        print(f"Filled {filled} / {len(result_df)} windows")
        print("\nFirst 10 scheduled targets:")
        print(result_df[["Target", "start", "stop", "Visibility"]].head(10))
    else:
        print("Scheduling returned False")
        filled = result_df[(result_df["Target"].notna()) & (result_df["Target"] != "") & (result_df["Target"] != "No target")].shape[0]
        unfilled = result_df[(result_df["Target"].isna()) | (result_df["Target"] == "") | (result_df["Target"] == "No target")].shape[0]
        print(f"Filled: {filled} / {len(result_df)} windows")
        print(f"Unfilled: {unfilled} windows")
        print("\nAll scheduled targets:")
        print(result_df[["Target", "start", "stop", "Visibility"]])
