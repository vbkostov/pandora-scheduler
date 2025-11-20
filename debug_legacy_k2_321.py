#!/usr/bin/env python3
"""Debug script to see what legacy does for K2-321."""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from astropy.time import Time

# Add legacy src to path
repo_root = Path(__file__).parent
legacy_path = repo_root / "src" / "pandorascheduler"
sys.path.insert(0, str(legacy_path))

import helper_codes_aux

# Set up data paths
PACKAGEDIR = str(legacy_path)
DATA_DIR = str(legacy_path / "data")

# K2-321 visit parameters
target_name = "K2-321"
visit_start = datetime(2026, 2, 10, 19, 25, 0)
visit_stop = datetime(2026, 2, 11, 17, 49, 0)

print("=" * 80)
print(f"Testing LEGACY occultation scheduling for {target_name}")
print(f"Visit: {visit_start} to {visit_stop}")
print("=" * 80)

# Load visibility and find occultation windows (same as rework would)
vis_file = Path(DATA_DIR) / "aux_targets" / target_name / f"Visibility for {target_name}.csv"
vis_df = pd.read_csv(vis_file)

vis_times_mjd = vis_df["Time(MJD_UTC)"].to_numpy()
vis_times_dt = Time(vis_times_mjd, format="mjd", scale="utc").to_datetime()
visibility_flags = vis_df["Visible"].to_numpy()

mask = (vis_times_dt >= visit_start) & (vis_times_dt <= visit_stop)
visit_times = vis_times_dt[mask]
visit_flags = visibility_flags[mask]

# Round times
def round_to_nearest_second(timestamp):
    if timestamp.microsecond >= 500_000:
        return timestamp + timedelta(seconds=1) - timedelta(microseconds=timestamp.microsecond)
    return timestamp - timedelta(microseconds=timestamp.microsecond)

visit_times = np.vectorize(round_to_nearest_second)(visit_times)

# Find visibility changes
if len(visit_flags) > 1:
    changes = np.where(visit_flags[:-1] != visit_flags[1:])[0]
    
    # Build occultation windows (legacy style)
    oc_starts = []
    oc_stops = []
    
    if not visit_flags[-1]:
        changes = changes.tolist()
        changes.append(len(visit_times) - 2)
        changes = np.array(changes)
    
    if not visit_flags[0]:
        oc_starts.append(visit_times[0])
        oc_stops.append(visit_times[changes[0]])
    
    for v in range(len(changes) - 1):
        if not visit_flags[changes[v] + 1]:
            oc_starts.append(visit_times[changes[v] + 1])
            oc_stops.append(visit_times[changes[v + 1]])
    
    print(f"\nOccultation windows: {len(oc_starts)}")
    
    # Break long sequences
    occultation_sequence_limit = timedelta(minutes=31)
    start_tmp, stop_tmp = [], []
    for ii in range(len(oc_stops)):
        ranges = helper_codes_aux.break_long_sequences(oc_starts[ii], oc_stops[ii], occultation_sequence_limit)
        if len(ranges) > 1:
            for jj in range(len(ranges)):
                start_tmp.append(ranges[jj][0])
                stop_tmp.append(ranges[jj][1])
        else:
            start_tmp.append(oc_starts[ii])
            stop_tmp.append(oc_stops[ii])
    oc_starts, oc_stops = start_tmp, stop_tmp
    
    print(f"After breaking: {len(oc_starts)} windows")
    
    # Convert to MJD
    starts_mjd = Time(oc_starts, format="datetime").to_value("mjd")
    stops_mjd = Time(oc_stops, format="datetime").to_value("mjd")
    
    # Create schedule DataFrame
    e_sched = [
        ["", datetime.strftime(oc_starts[s], "%Y-%m-%dT%H:%M:%SZ"), 
         datetime.strftime(oc_stops[s], "%Y-%m-%dT%H:%M:%SZ"), "", ""]
        for s in range(len(oc_starts))
    ]
    o_df = pd.DataFrame(e_sched, columns=["Target", "start", "stop", "RA", "DEC"])
    
    # Load occultation list
    occ_path = Path(DATA_DIR) / "occultation-standard_targets.csv"
    o_list = pd.read_csv(occ_path)
    v_names = o_list["Star Name"].to_numpy()
    
    path_ = str(Path(DATA_DIR) / "aux_targets")
    
    print(f"\nCalling legacy schedule_occultation_targets...")
    print(f"  Windows: {len(starts_mjd)}")
    print(f"  Candidates: {len(v_names)}")
    
    # Call legacy function
    o_df, d_flag = helper_codes_aux.schedule_occultation_targets(
        v_names,
        starts_mjd,
        stops_mjd,
        path_,
        o_df,
        o_list,
        "occ list",
    )
    
    print(f"\nResult flag: {d_flag}")
    filled = o_df[(o_df["Target"].notna()) & (o_df["Target"] != "") & (o_df["Target"] != "No target")].shape[0]
    print(f"Filled: {filled} / {len(o_df)} windows")
    
    print("\nAll targets:")
    print(o_df[["Target", "start", "stop", "Visibility"]])
