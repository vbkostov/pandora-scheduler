"""Utility to compare legacy and rework minute sets for a target."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time


def _truncate_to_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)


def compare_target(star_dir: Path, planet_subdir: Path) -> None:
    star_df = pd.read_csv(star_dir / f"Visibility for {star_dir.name}.csv")
    planet_df = pd.read_csv(planet_subdir / f"Visibility for {planet_subdir.name}.csv")

    vis_mask = star_df["Visible"].to_numpy(dtype=float)
    saa_mask = star_df["SAA_Crossing"].to_numpy(dtype=float)
    vis_mjd = star_df["Time(MJD_UTC)"].to_numpy(dtype=float)

    t_mjd = Time(vis_mjd, format="mjd", scale="utc")
    t_iso = Time(t_mjd.iso, format="iso", scale="utc")
    dt_iso = t_iso.to_value("datetime")

    legacy_vis = [dt for dt, flag in zip(dt_iso, vis_mask) if flag == 1.0]
    legacy_saa = [dt for dt, flag in zip(dt_iso, saa_mask) if flag == 1.0]

    rework_time = Time(vis_mjd, format="mjd", scale="utc")
    rework_iso = Time(rework_time.iso, format="iso", scale="utc")
    rework_dt = [dt for dt in rework_iso.to_value("datetime")]

    rework_vis = [dt for dt, flag in zip(rework_dt, vis_mask) if flag == 1.0]
    rework_saa = [dt for dt, flag in zip(rework_dt, saa_mask) if flag == 1.0]

    start_values = planet_df["Transit_Start"].to_numpy(dtype=float)
    stop_values = planet_df["Transit_Stop"].to_numpy(dtype=float)

    start_dt = [
        _truncate_to_minute(dt)
        for dt in Time(start_values, format="mjd", scale="utc").to_datetime(timezone=None)
    ]
    stop_dt = [
        _truncate_to_minute(dt)
        for dt in Time(stop_values, format="mjd", scale="utc").to_datetime(timezone=None)
    ]

    legacy_cov = []
    rework_cov = []
    for start, stop in zip(start_dt, stop_dt):
        minute_range = pd.date_range(start, stop, freq="min").to_pydatetime()
        total = len(minute_range)
        if total == 0:
            legacy_cov.append(0.0)
            rework_cov.append(0.0)
            continue
        minute_set = set(minute_range)
        legacy_cov.append(len(minute_set.intersection(legacy_vis)) / total)
        rework_cov.append(len(minute_set.intersection(rework_vis)) / total)

    legacy_cov = np.asarray(legacy_cov, dtype=float)
    rework_cov = np.asarray(rework_cov, dtype=float)

    print(f"Legacy minute sample: {legacy_vis[:3]}")
    print(f"Rework minute sample: {rework_vis[:3]}")
    print("max |legacy - rework|:", float(np.max(np.abs(legacy_cov - rework_cov))))

    diff_indices = np.where(np.abs(legacy_cov - rework_cov) > 1e-9)[0]
    print("Differing indices:", diff_indices[:10])
    if diff_indices.size:
        idx = diff_indices[0]
        minute_range = pd.date_range(start_dt[idx], stop_dt[idx], freq="min").to_pydatetime()
        minute_set = set(minute_range)
        legacy_overlap = minute_set.intersection(legacy_vis)
        rework_overlap = minute_set.intersection(rework_vis)
        print(f"Example index {idx} legacy coverage {legacy_cov[idx]} rework {rework_cov[idx]}")
        print("Legacy-only minutes:", sorted(legacy_overlap - rework_overlap)[:10])
        print("Rework-only minutes:", sorted(rework_overlap - legacy_overlap)[:10])

    legacy_saa_overlap = len(set(legacy_saa))
    rework_saa_overlap = len(set(rework_saa))
    print("Legacy SAA samples:", legacy_saa[:3])
    print("Rework SAA samples:", rework_saa[:3])
    print("SAA minute counts legacy vs rework:", legacy_saa_overlap, rework_saa_overlap)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("star_dir", type=Path)
    parser.add_argument("planet", type=str)
    args = parser.parse_args()

    star_dir = args.star_dir
    planet_dir = star_dir / args.planet
    compare_target(star_dir, planet_dir)


if __name__ == "__main__":
    main()
