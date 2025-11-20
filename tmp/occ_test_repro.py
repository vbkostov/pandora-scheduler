from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import shutil
import tempfile

import pandas as pd
from astropy.time import Time

from pandorascheduler_rework import observation_utils, science_calendar


def write_visibility(base_dir: Path, subpath: str, name: str, flags: list[int]) -> None:
    target_dir = base_dir.joinpath(subpath, name)
    target_dir.mkdir(parents=True, exist_ok=True)
    start = datetime(2026, 1, 2, 0, 0, 0)
    times = [start + timedelta(minutes=value) for value in range(len(flags))]
    mjd_times = Time(times, scale="utc").to_value("mjd")
    pd.DataFrame({"Time(MJD_UTC)": mjd_times, "Visible": flags}).to_csv(
        target_dir / f"Visibility for {name}.csv",
        index=False,
    )


def write_planet_visibility(base_dir: Path) -> None:
    start = datetime(2026, 1, 2, 0, 0, 0)
    directory = base_dir / "targets" / "StarTwo" / "StarTwo b"
    directory.mkdir(parents=True, exist_ok=True)
    transit_start = Time([start + timedelta(minutes=20)], scale="utc").to_value("mjd")
    transit_stop = Time([start + timedelta(minutes=40)], scale="utc").to_value("mjd")
    pd.DataFrame(
        {
            "Transit_Start": transit_start,
            "Transit_Stop": transit_stop,
            "Transit_Coverage": [0.75],
            "SAA_Overlap": [0.0],
        }
    ).to_csv(directory / "Visibility for StarTwo b.csv", index=False)


def main() -> None:
    tmp_root = Path(tempfile.mkdtemp(prefix="occ-repro-"))
    data_dir = tmp_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    primary_flags = [1] * 30 + [0] * 60 + [1] * 90
    occ_a_flags = [0] * 30 + [1] * 31 + [0] * (180 - 61)
    occ_b_flags = [0] * 61 + [1] * 29 + [0] * (180 - 90)

    write_visibility(data_dir, "targets", "StarTwo", primary_flags)
    write_planet_visibility(data_dir)
    write_visibility(data_dir, "targets/StarTwo", "StarTwo b", [1] * 180)
    write_visibility(data_dir, "aux_targets", "OccA", occ_a_flags)
    write_visibility(data_dir, "aux_targets", "OccB", occ_b_flags)

    pd.DataFrame(
        [
            {"Planet Name": "StarTwo b", "Star Name": "StarTwo", "RA": 15.0, "DEC": -10.0}
        ]
    ).to_csv(data_dir / "exoplanet_targets.csv", index=False)

    occ_catalog = pd.DataFrame(
        [
            {"Star Name": "OccA", "RA": 30.0, "DEC": 10.0},
            {"Star Name": "OccB", "RA": 35.0, "DEC": 12.0},
        ]
    )
    occ_catalog.to_csv(data_dir / "occultation-standard_targets.csv", index=False)
    occ_catalog.to_csv(data_dir / "aux_list_new.csv", index=False)

    observation_utils.DATA_ROOTS = [data_dir]

    schedule_path = tmp_root / "schedule.csv"
    pd.DataFrame(
        [
            {
                "Target": "StarTwo b",
                "Observation Start": "2026-01-02 00:00:00",
                "Observation Stop": "2026-01-02 03:00:00",
                "Transit Coverage": 0.8,
                "SAA Overlap": 0.0,
                "Schedule Factor": 0.75,
                "Quality Factor": 0.7,
                "Comments": "",
            }
        ]
    ).to_csv(schedule_path, index=False)

    inputs = science_calendar.ScienceCalendarInputs(schedule_csv=schedule_path, data_dir=data_dir)
    config = science_calendar.ScienceCalendarConfig(
        visit_limit=1,
        prioritise_occultations_by_slew=True,
    )

    builder = science_calendar._ScienceCalendarBuilder(inputs, config)
    row = builder.schedule.iloc[0]
    start = science_calendar._parse_datetime(row["Observation Start"])
    stop = science_calendar._parse_datetime(row["Observation Stop"])
    target, star = science_calendar._normalise_target_name(row["Target"])
    visibility = science_calendar._read_visibility(builder.data_dir / "targets" / star, star)
    times, flags = science_calendar._extract_visibility_segment(
        visibility,
        start,
        stop,
        builder.config.min_sequence_minutes,
    )
    changes = science_calendar._visibility_change_indices(flags)
    oc_starts, oc_stops, augmented = science_calendar._occultation_windows(times, flags, changes)
    print("Occultation windows:", list(zip(oc_starts, oc_stops)))

    expanded_starts: list[datetime] = []
    expanded_stops: list[datetime] = []
    for oc_start, oc_stop in zip(oc_starts, oc_stops):
        for seg_start, seg_stop in observation_utils.break_long_sequences(
            oc_start,
            oc_stop,
            builder.occultation_limit,
        ):
            expanded_starts.append(seg_start)
            expanded_stops.append(seg_stop)

    print("Expanded windows:", list(zip(expanded_starts, expanded_stops)))

    occ_list = pd.read_csv(data_dir / "occultation-standard_targets.csv")
    occ_df = pd.DataFrame(
        {
            "Target": ["" for _ in expanded_starts],
            "start": [value.strftime("%Y-%m-%dT%H:%M:%SZ") for value in expanded_starts],
            "stop": [value.strftime("%Y-%m-%dT%H:%M:%SZ") for value in expanded_stops],
            "RA": ["" for _ in expanded_starts],
            "DEC": ["" for _ in expanded_starts],
        }
    )

    starts_mjd = Time(expanded_starts, format="datetime", scale="utc").to_value("mjd")
    stops_mjd = Time(expanded_stops, format="datetime", scale="utc").to_value("mjd")

    schedule_copy, flag = observation_utils.schedule_occultation_targets(
        occ_list["Star Name"].to_numpy(),
        starts_mjd,
        stops_mjd,
        start,
        stop,
        str(data_dir / "aux_targets"),
        occ_df.copy(),
        occ_list,
        "occ list",
    )
    print("Helper flag:", flag)
    print(schedule_copy)

    shutil.rmtree(tmp_root)


if __name__ == "__main__":
    main()
