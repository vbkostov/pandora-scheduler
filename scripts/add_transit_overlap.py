#!/usr/bin/env python3
"""
Add Transit_Overlap column to visibility files using legacy transits module.
This is much faster than running the full scheduler.
"""
import sys
from pathlib import Path
import importlib.util
import pandas as pd

# Import transits module directly
transits_path = Path(__file__).parent.parent / "src" / "pandorascheduler" / "transits.py"
spec = importlib.util.spec_from_file_location("transits", transits_path)
transits = importlib.util.module_from_spec(spec)
sys.modules["transits"] = transits
spec.loader.exec_module(transits)

PACKAGEDIR = Path(__file__).parent.parent / "src" / "pandorascheduler"
DATA_DIR = PACKAGEDIR / "data"

def main():
    print("=" * 80)
    print("ADDING TRANSIT_OVERLAP TO VISIBILITY FILES")
    print("=" * 80)
    
    manifest_path = DATA_DIR / "exoplanet_targets.csv"
    targets = pd.read_csv(manifest_path)
    
    # Get unique star names
    star_names = targets['Star Name'].unique()
    
    print(f"\nProcessing {len(star_names)} star systems...")
    print("-" * 80)
    
    for i, star_name in enumerate(star_names):
        print(f"[{i+1}/{len(star_names)}] {star_name}...", end=" ", flush=True)
        
        try:
            transits.Transit_overlap(
                target_list="exoplanet_targets.csv",
                partner_list="exoplanet_targets.csv",
                star_name=star_name
            )
            print("✅")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 80)
    print("TRANSIT_OVERLAP COMPLETE")
    print("=" * 80)
    print("\nRun validation to verify:")
    print("  python scripts/validate_visibility_data.py")

if __name__ == "__main__":
    main()
