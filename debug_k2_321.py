#!/usr/bin/env python3
"""Debug script to trace K2-321 visit processing."""

import sys
from pathlib import Path
from datetime import datetime
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

# K2-321 visit parameters from schedule
target_name = "K2-321"
visit_start = datetime(2026, 2, 10, 19, 25, 0)
visit_stop = datetime(2026, 2, 11, 17, 49, 0)

print(f"Debugging K2-321 visit: {visit_start} to {visit_stop}")
print("=" * 80)

# Check if visibility file exists
vis_file = data_dir / "aux_targets" / target_name / f"Visibility for {target_name}.csv"
print(f"\nVisibility file: {vis_file}")
print(f"Exists: {vis_file.exists()}")

if vis_file.exists():
    # Load visibility data
    vis_df = pd.read_csv(vis_file)
    print(f"Visibility data shape: {vis_df.shape}")
    print(f"Columns: {vis_df.columns.tolist()}")
    
    # Convert to datetime
    vis_times_mjd = vis_df["Time(MJD_UTC)"].to_numpy()
    vis_times_dt = Time(vis_times_mjd, format="mjd", scale="utc").to_datetime()
    visibility_flags = vis_df["Visible"].to_numpy()
    
    # Extract segment within visit window
    mask = (vis_times_dt >= visit_start) & (vis_times_dt <= visit_stop)
    visit_times = vis_times_dt[mask]
    visit_flags = visibility_flags[mask]
    
    print(f"\nSamples within visit window: {len(visit_times)}")
    print(f"Visible samples: {np.sum(visit_flags == 1)}")
    print(f"Non-visible samples: {np.sum(visit_flags == 0)}")
    
    # Find visibility changes
    if len(visit_flags) > 1:
        changes = np.where(visit_flags[:-1] != visit_flags[1:])[0]
        print(f"\nVisibility changes at indices: {changes}")
        
        if len(changes) > 0:
            # Identify occultation windows
            oc_starts = []
            oc_stops = []
            
            # Check if starts occluded
            if visit_flags[0] == 0:
                oc_starts.append(visit_times[0])
                if len(changes) > 0:
                    oc_stops.append(visit_times[changes[0]])
            
            # Process remaining changes
            for i in range(len(changes) - 1):
                if visit_flags[changes[i] + 1] == 0:
                    oc_starts.append(visit_times[changes[i] + 1])
                    oc_stops.append(visit_times[changes[i + 1]])
            
            # Check if ends occluded
            if visit_flags[-1] == 0 and len(changes) > 0:
                if visit_flags[changes[-1]] == 1:
                    oc_starts.append(visit_times[changes[-1] + 1])
                    oc_stops.append(visit_times[-1])
            
            print(f"\nOccultation windows found: {len(oc_starts)}")
            for i, (start, stop) in enumerate(zip(oc_starts, oc_stops)):
                duration = (stop - start).total_seconds() / 60
                print(f"  Window {i+1}: {start} to {stop} ({duration:.1f} min)")
        else:
            print("\nNo visibility changes - target fully visible or fully occluded")
            if visit_flags[0] == 1:
                print("Target is fully VISIBLE throughout visit")
            else:
                print("Target is fully OCCLUDED throughout visit")
    else:
        print("\nInsufficient visibility samples")

# Now test the rework's occultation scheduling
print("\n" + "=" * 80)
print("Testing rework occultation scheduling")
print("=" * 80)

# Try to find occultation targets using rework logic
occ_list_path = data_dir / "occultation-standard_targets.csv"
print(f"\nOccultation list: {occ_list_path}")
print(f"Exists: {occ_list_path.exists()}")

if occ_list_path.exists():
    occ_list = pd.read_csv(occ_list_path)
    print(f"Occultation targets available: {len(occ_list)}")
    print(f"First few targets: {occ_list['Star Name'].head(10).tolist()}")
