import shutil
import tempfile
from pathlib import Path

from pandorascheduler_rework import observation_utils, science_calendar


def main():
    repo_root = Path(__file__).resolve().parents[1]
    legacy_package_dir = repo_root / "src" / "pandorascheduler"

    tmp_root = Path(tempfile.mkdtemp(prefix="xml-debug-"))
    legacy_copy = tmp_root / "legacy"
    shutil.copytree(legacy_package_dir, legacy_copy)

    schedule_src = repo_root / "comparison_outputs" / "rework" / (
        "Pandora_Schedule_0.8_0.0_0.2_2026-02-05_to_2026-02-06.csv"
    )
    legacy_schedule = legacy_copy / "data" / schedule_src.name
    shutil.copy2(schedule_src, legacy_schedule)

    observation_utils.DATA_ROOTS = [legacy_copy / "data"]
    inputs = science_calendar.ScienceCalendarInputs(
        schedule_csv=legacy_schedule,
        data_dir=legacy_copy / "data",
    )
    builder = science_calendar._ScienceCalendarBuilder(
        inputs, science_calendar.ScienceCalendarConfig(visit_limit=None)
    )

    row = builder.schedule.iloc[1]
    print("Row target:", row["Target"])
    start = science_calendar._parse_datetime(row["Observation Start"])
    stop = science_calendar._parse_datetime(row["Observation Stop"])
    target, star = science_calendar._normalise_target_name(str(row["Target"]))
    print("Normalised target/star:", target, star)
    planet_row = science_calendar._lookup_planet_row(builder.target_catalog, target)
    print("Planet row present:", planet_row is not None)
    print("Transit flag:", science_calendar._is_transit_entry(row))

    if planet_row is not None and science_calendar._is_transit_entry(row):
        vis_dir = builder.data_dir / "targets" / star
        vis_name = star
        target_info = planet_row
    else:
        vis_dir = builder.data_dir / "aux_targets" / target
        vis_name = target
        target_info = science_calendar._lookup_auxiliary_row(builder.aux_catalog, target)
    print("Visibility dir:", vis_dir)

    vis = science_calendar._read_visibility(vis_dir, vis_name)
    print("Visibility shape:", vis.shape if vis is not None else None)

    if vis is None or start is None or stop is None:
        raise RuntimeError("Missing visibility data or schedule timestamps")

    times, flags = science_calendar._extract_visibility_segment(
        vis,
        start,
        stop,
        builder.config.min_sequence_minutes,
    )
    print("Visibility samples:", len(times))
    changes = science_calendar._visibility_change_indices(flags)
    oc_starts, oc_stops, augmented = science_calendar._occultation_windows(
        times,
        flags,
        changes,
    )
    print("Occultation windows:")
    for s, e in zip(oc_starts, oc_stops):
        print("  ", s, "to", e)

    if target_info is not None:
        try:
            ra_value = float(target_info.iloc[0]["RA"])
            dec_value = float(target_info.iloc[0]["DEC"])
        except Exception:
            ra_value = float("nan")
            dec_value = float("nan")
    else:
        ra_value = float("nan")
        dec_value = float("nan")

    print("Reference RA/DEC:", ra_value, dec_value)

    occ_result = builder._find_occultation_target(
        oc_starts,
        oc_stops,
        start,
        stop,
        ra_value,
        dec_value,
    )
    print("Occultation selection success:", occ_result is not None)
    if occ_result:
        df, flag = occ_result
        print("Helper flag:", flag)
        print(df)

    print("Temp data dir:", legacy_copy / "data")


if __name__ == "__main__":
    main()
