#!/usr/bin/env python3
"""
Regenerate legacy visibility files using the legacy transits.py module.
This script properly calls the legacy functions with correct parameters.
"""
import sys
import os
from pathlib import Path

# Add legacy package to path
legacy_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(legacy_path))

# Import transits module directly to avoid broken package __init__
import importlib.util

transits_path = Path(__file__).parent.parent / "src" / "pandorascheduler" / "transits.py"
spec = importlib.util.spec_from_file_location("transits", transits_path)
transits = importlib.util.module_from_spec(spec)
sys.modules["transits"] = transits
spec.loader.exec_module(transits)

import pandas as pd

PACKAGEDIR = Path(__file__).parent.parent / "src" / "pandorascheduler"
DATA_DIR = PACKAGEDIR / "data"

def main():
    print("=" * 80)
    print("REGENERATING LEGACY VISIBILITY FILES")
    print("=" * 80)
    
    # Parameters from legacy system
    sun_block = 91.0
    moon_block = 25.0
    earth_block = 63.0  # From comment in transits.py: "based on worst case orbital alt of 450km should be 63 deg"
    
    # Full mission lifetime (not just test window)
    obs_start = "2025-04-25 00:00:00"
    obs_stop = "2026-04-25 00:00:00"
    
    gmat_file = "Pandora-600km-withoutdrag-20251018.txt"
    obs_name = "Pandora"
    save_pth = str(DATA_DIR / "targets") + "/"
    targ_list = str(DATA_DIR / "exoplanet_targets.csv")
    
    print(f"\nParameters:")
    print(f"  Sun block:    {sun_block}°")
    print(f"  Moon block:   {moon_block}°")
    print(f"  Earth block:  {earth_block}°")
    print(f"  Start:        {obs_start}")
    print(f"  Stop:         {obs_stop}")
    print(f"  GMAT file:    {gmat_file}")
    print(f"  Save path:    {save_pth}")
    print(f"  Target list:  {targ_list}")
    print()
    
    # Step 1: Generate star visibility
    print("STEP 1: Generating star visibility...")
    print("-" * 80)
    transits.star_vis(
        sun_block=sun_block,
        moon_block=moon_block,
        earth_block=earth_block,
        obs_start=obs_start,
        obs_stop=obs_stop,
        gmat_file=gmat_file,
        obs_name=obs_name,
        save_pth=save_pth,
        targ_list=targ_list
    )
    print("✅ Star visibility complete\n")
    
    # Step 2: Generate planet transits
    print("STEP 2: Generating planet transits...")
    print("-" * 80)
    targets = pd.read_csv(targ_list)
    
    for i, row in targets.iterrows():
        planet_name = row['Planet Name']
        star_name = row['Star Name']
        print(f"  [{i+1}/{len(targets)}] {planet_name}...", end=" ", flush=True)
        
        try:
            transits.transit_timing(
                target_list="exoplanet_targets.csv",  # Relative to PACKAGEDIR/data
                planet_name=planet_name,
                star_name=star_name
            )
            print("✅")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("✅ Planet transits complete\n")
    
    # Step 3: Add SAA overlap
    print("STEP 3: Adding SAA overlap...")
    print("-" * 80)
    for i, row in targets.iterrows():
        planet_name = row['Planet Name']
        star_name = row['Star Name']
        print(f"  [{i+1}/{len(targets)}] {planet_name}...", end=" ", flush=True)
        
        try:
            transits.SAA_overlap(
                planet_name=planet_name,
                star_name=star_name
            )
            print("✅")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("✅ SAA overlap complete\n")
    
    # Step 4: Add transit overlap
    print("STEP 4: Adding transit overlap...")
    print("-" * 80)
    
    # Get unique star names
    star_names = targets['Star Name'].unique()
    
    for i, star_name in enumerate(star_names):
        print(f"  [{i+1}/{len(star_names)}] {star_name}...", end=" ", flush=True)
        
        try:
            transits.Transit_overlap(
                target_list="exoplanet_targets.csv",
                partner_list="exoplanet_targets.csv",
                star_name=star_name
            )
            print("✅")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("✅ Transit overlap complete\n")
    
    print("=" * 80)
    print("REGENERATION COMPLETE")
    print("=" * 80)
    print("\nRun validation script to verify:")
    print("  python scripts/validate_visibility_data.py")

if __name__ == "__main__":
    main()
