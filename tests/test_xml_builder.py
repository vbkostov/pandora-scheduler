from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from astropy.time import Time
import xml.etree.ElementTree as ET

from pandorascheduler_rework import observation_utils
from pandorascheduler_rework import science_calendar


def _write_visibility(directory: Path, name: str, times: list[datetime], flags: list[int]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    mjd_times = Time(times, scale="utc").to_value("mjd")
    pd.DataFrame({"Time(MJD_UTC)": mjd_times, "Visible": flags}).to_csv(
        directory / f"Visibility for {name}.csv", index=False
    )


def _write_planet_visibility(directory: Path, name: str, start: datetime, stop: datetime) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    transit_start = Time([start], scale="utc").to_value("mjd")
    transit_stop = Time([stop], scale="utc").to_value("mjd")
    pd.DataFrame(
        {
            "Transit_Start": transit_start,
            "Transit_Stop": transit_stop,
            "Transit_Coverage": [0.75],
            "SAA_Overlap": [0.1],
        }
    ).to_csv(directory / f"Visibility for {name}.csv", index=False)


def test_generate_science_calendar_with_occultation(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    start = datetime(2026, 1, 1, 0, 0, 0)
    times = [start + timedelta(minutes=value) for value in range(0, 180)]
    visibility_flags = [1] * 60 + [0] * 60 + [1] * 60

    _write_visibility(data_dir / "targets" / "StarOne", "StarOne", times, visibility_flags)
    _write_planet_visibility(
        data_dir / "targets" / "StarOne" / "StarOne b",
        "StarOne b",
        start + timedelta(minutes=30),
        start + timedelta(minutes=90),
    )
    _write_visibility(data_dir / "aux_targets" / "OccStar", "OccStar", times, [1] * len(times))

    pd.DataFrame(
        [
            {
                "Planet Name": "StarOne b",
                "Star Name": "StarOne",
                "RA": 10.0,
                "DEC": -20.0,
            }
        ]
    ).to_csv(data_dir / "exoplanet_targets.csv", index=False)

    pd.DataFrame(
        [
            {
                "Star Name": "OccStar",
                "RA": 30.0,
                "DEC": 15.0,
            }
        ]
    ).to_csv(data_dir / "aux_list_new.csv", index=False)

    pd.DataFrame(
        [
            {
                "Star Name": "OccStar",
                "RA": 30.0,
                "DEC": 15.0,
            }
        ]
    ).to_csv(data_dir / "occultation-standard_targets.csv", index=False)

    schedule_df = pd.DataFrame(
        [
            {
                "Target": "StarOne b",
                "Observation Start": "2026-01-01 00:00:00",
                "Observation Stop": "2026-01-01 03:00:00",
                "Transit Coverage": 0.75,
                "SAA Overlap": 0.1,
                "Schedule Factor": 0.9,
                "Quality Factor": 0.85,
                "Comments": "",
            }
        ]
    )
    schedule_path = tmp_path / "schedule.csv"
    schedule_df.to_csv(schedule_path, index=False)

    monkeypatch.setattr(observation_utils, "DATA_ROOTS", [data_dir])

    inputs = science_calendar.ScienceCalendarInputs(schedule_csv=schedule_path, data_dir=data_dir)
    config = science_calendar.ScienceCalendarConfig(
        visit_limit=1,
        prioritise_occultations_by_slew=True,
        created_timestamp=datetime(2025, 11, 17, 18, 10, 32),
    )
    output_path = tmp_path / "calendar.xml"

    generated_path = science_calendar.generate_science_calendar(
        inputs, output_path=output_path, config=config
    )

    assert generated_path == output_path

    xml_text = generated_path.read_text(encoding="utf-8")
    assert "<Target>StarOne b</Target>" in xml_text
    assert "<Target>OccStar</Target>" in xml_text

    # Ensure the metadata reflects the configured minimum sequence pruning value
    assert "Removed_Sequences_Shorter_Than_min=\"5\"" in xml_text
    assert "Created=\"2025-11-17 18:10:32\"" in xml_text


def test_generate_science_calendar_splits_long_occultations(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    start = datetime(2026, 1, 2, 0, 0, 0)
    total_minutes = 180
    times = [start + timedelta(minutes=value) for value in range(total_minutes)]

    primary_flags = [1] * 30 + [0] * 60 + [1] * 90
    _write_visibility(data_dir / "targets" / "StarTwo", "StarTwo", times, primary_flags)
    _write_planet_visibility(
        data_dir / "targets" / "StarTwo" / "StarTwo b",
        "StarTwo b",
        start + timedelta(minutes=20),
        start + timedelta(minutes=40),
    )

    occ_a_flags = [0] * 30 + [1] * 32 + [0] * (total_minutes - 62)
    occ_b_flags = [0] * 60 + [1] * 31 + [0] * (total_minutes - 91)
    _write_visibility(data_dir / "aux_targets" / "OccA", "OccA", times, occ_a_flags)
    _write_visibility(data_dir / "aux_targets" / "OccB", "OccB", times, occ_b_flags)

    pd.DataFrame(
        [
            {"Planet Name": "StarTwo b", "Star Name": "StarTwo", "RA": 15.0, "DEC": -10.0}
        ]
    ).to_csv(data_dir / "exoplanet_targets.csv", index=False)

    pd.DataFrame(
        [
            {"Star Name": "OccA", "RA": 30.0, "DEC": 10.0},
            {"Star Name": "OccB", "RA": 35.0, "DEC": 12.0},
        ]
    ).to_csv(data_dir / "occultation-standard_targets.csv", index=False)

    pd.DataFrame(
        [
            {"Star Name": "OccA", "RA": 30.0, "DEC": 10.0},
            {"Star Name": "OccB", "RA": 35.0, "DEC": 12.0},
        ]
    ).to_csv(data_dir / "aux_list_new.csv", index=False)

    schedule_df = pd.DataFrame(
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
    )
    schedule_path = tmp_path / "schedule_long_occ.csv"
    schedule_df.to_csv(schedule_path, index=False)

    monkeypatch.setattr(observation_utils, "DATA_ROOTS", [data_dir])

    inputs = science_calendar.ScienceCalendarInputs(schedule_csv=schedule_path, data_dir=data_dir)
    config = science_calendar.ScienceCalendarConfig(
        visit_limit=1,
        prioritise_occultations_by_slew=True,
    )
    output_path = tmp_path / "calendar_long_occ.xml"

    science_calendar.generate_science_calendar(inputs, output_path=output_path, config=config)

    xml_text = output_path.read_text(encoding="utf-8")
    assert "<Target>StarTwo b</Target>" in xml_text
    assert "<Target>OccA</Target>" in xml_text
    assert "<Target>OccB</Target>" in xml_text


def test_observation_sequence_uses_manifest_star_roi_method():
    visit = ET.Element("Visit")
    start = datetime(2026, 1, 3, 0, 0, 0)
    stop = start + timedelta(minutes=10)

    targ_info = pd.DataFrame(
        [
            {
                "Star Name": "DemoStar",
                "Planet Name": "DemoStar b",
                "StarRoiDetMethod": 1,
                "RA": 123.0,
                "DEC": -45.0,
                "VDA_StarRoiDetMethod": "SET_BY_TARGET_DEFINITION_FILE",
            }
        ]
    )

    observation_utils.observation_sequence(
        visit,
        "001",
        "DemoStar b",
        "0",
        start,
        stop,
        123.0,
        -45.0,
        targ_info,
    )

    star_roi = visit.find(
        "Observation_Sequence/Payload_Parameters/AcquireVisCamScienceData/StarRoiDetMethod"
    )
    assert star_roi is not None
    assert star_roi.text == "1"


def test_visit_id_formatting_matches_legacy_quirk(tmp_path, monkeypatch):
    """Test that Visit IDs use legacy's quirky padding formula.
    
    Legacy bug: Uses len(str(i)) instead of len(str(i+1)), producing 5 digits for 2-digit visits.
    For i=9: "0" * (4 - len(str(9))) + str(10) = "000" + "10" = "00010"
    
    This test ensures we match that behavior to maintain XML parity.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    start = datetime(2026, 1, 1, 0, 0, 0)
    times = [start + timedelta(hours=i) for i in range(24)]
    _write_visibility(data_dir / "aux_targets" / "TestStar", "TestStar", times, [1] * len(times))

    pd.DataFrame([{"Star Name": "TestStar", "RA": 10.0, "DEC": -20.0}]).to_csv(
        data_dir / "aux_list_new.csv", index=False
    )
    pd.DataFrame([{"Star Name": "TestStar", "RA": 10.0, "DEC": -20.0}]).to_csv(
        data_dir / "exoplanet_targets.csv", index=False
    )
    pd.DataFrame([{"Star Name": "TestStar", "RA": 10.0, "DEC": -20.0}]).to_csv(
        data_dir / "occultation-standard_targets.csv", index=False
    )

    # Create schedule with 10 visits to test the formatting transition
    schedule_rows = []
    for visit_num in range(1, 11):
        schedule_rows.append({
            "Target": "TestStar",
            "Observation Start": (start + timedelta(hours=visit_num * 2)).strftime("%Y-%m-%d %H:%M:%S"),
            "Observation Stop": (start + timedelta(hours=visit_num * 2 + 1)).strftime("%Y-%m-%d %H:%M:%S"),
            "Transit Coverage": None,
            "SAA Overlap": None,
            "Schedule Factor": 0.9,
            "Quality Factor": 0.85,
            "Comments": "",
        })
    
    schedule_df = pd.DataFrame(schedule_rows)
    schedule_path = tmp_path / "schedule.csv"
    schedule_df.to_csv(schedule_path, index=False)

    monkeypatch.setattr(observation_utils, "DATA_ROOTS", [data_dir])

    inputs = science_calendar.ScienceCalendarInputs(schedule_csv=schedule_path, data_dir=data_dir)
    config = science_calendar.ScienceCalendarConfig(visit_limit=10)
    output_path = tmp_path / "calendar.xml"

    science_calendar.generate_science_calendar(inputs, output_path=output_path, config=config)

    xml_text = output_path.read_text(encoding="utf-8")
    
    # Verify the formatting for visits 1-10
    assert "<ID>0001</ID>" in xml_text  # Visit 1: "0" * (4-0) + "1" = "00001" (5 digits)
    assert "<ID>0002</ID>" in xml_text  # Visit 2: "0" * (4-0) + "2" = "00002" (5 digits)
    assert "<ID>0009</ID>" in xml_text  # Visit 9: "0" * (4-0) + "9" = "00009" (5 digits)
    assert "<ID>00010</ID>" in xml_text  # Visit 10: "0" * (4-1) + "10" = "00010" (5 digits!)
    
    # Ensure we DON'T have the "correct" 4-digit format
    assert "<ID>0010</ID>" not in xml_text
