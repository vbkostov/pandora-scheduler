from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time

from pandorascheduler_rework.config import PandoraSchedulerConfig
import pandorascheduler_rework.observation_utils as observation_utils
from pandorascheduler_rework.observation_utils import schedule_occultation_targets
from pandorascheduler_rework.pipeline import _append_windowed_occultation_targets
from pandorascheduler_rework.science_calendar import (
    ScienceCalendarInputs,
    _ScienceCalendarBuilder,
)


def _write_minimal_catalogs(data_dir):
    pd.DataFrame(
        [{"Planet Name": "Test b", "Star Name": "Test", "RA": 0.0, "DEC": 0.0}]
    ).to_csv(data_dir / "exoplanet_targets.csv", index=False)
    pd.DataFrame([{ "Star Name": "HD_29037", "RA": 68.0, "DEC": 32.0,
                    "Number of Hours Requested": 24.0 }]).to_csv(
        data_dir / "occultation-standard_targets.csv", index=False
    )
    pd.DataFrame([{ "Star Name": "HD_29037", "RA": 68.0, "DEC": 32.0 }]).to_csv(
        data_dir / "all_targets.csv", index=False
    )


def test_occultation_windows_split_and_restrict_target(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_minimal_catalogs(data_dir)

    schedule_path = tmp_path / "schedule.csv"
    pd.DataFrame(
        [{
            "Target": "Test b",
            "Observation Start": "2026-11-20 16:00:00",
            "Observation Stop": "2026-11-21 18:00:00",
        }]
    ).to_csv(schedule_path, index=False)

    windows_path = tmp_path / "occultation_windows.csv"
    pd.DataFrame(
        [{
            "Target": "HD_29037",
            "Obs Window Start": "2026-11-20 17:00:00",
            "Obs Window Stop": "2026-11-21 17:00:00",
        }]
    ).to_csv(windows_path, index=False)

    config = PandoraSchedulerConfig(
        window_start=datetime(2026, 11, 20),
        window_end=datetime(2026, 11, 22),
        occultation_windows_csv=windows_path,
    )
    builder = _ScienceCalendarBuilder(
        ScienceCalendarInputs(schedule_csv=schedule_path, data_dir=data_dir),
        config,
    )

    starts, stops = builder._split_occultation_intervals_at_window_boundaries(
        [datetime(2026, 11, 20, 16, 0)],
        [datetime(2026, 11, 21, 18, 0)],
    )

    assert list(zip(starts, stops)) == [
        (datetime(2026, 11, 20, 16, 0), datetime(2026, 11, 20, 17, 0)),
        (datetime(2026, 11, 20, 17, 0), datetime(2026, 11, 21, 17, 0)),
        (datetime(2026, 11, 21, 17, 0), datetime(2026, 11, 21, 18, 0)),
    ]
    assert not builder._occultation_target_allowed(
        "HD_29037", starts[0], stops[0]
    )
    assert builder._occultation_target_allowed(
        "HD_29037", starts[1], stops[1]
    )
    assert not builder._occultation_target_allowed(
        "HD_29037", starts[2], stops[2]
    )


def test_unconfigured_occultation_targets_remain_unrestricted(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_minimal_catalogs(data_dir)

    schedule_path = tmp_path / "schedule.csv"
    pd.DataFrame(
        [{"Target": "Test b", "Observation Start": "2026-11-20 00:00:00",
          "Observation Stop": "2026-11-21 00:00:00"}]
    ).to_csv(schedule_path, index=False)
    windows_path = tmp_path / "occultation_windows.csv"
    pd.DataFrame(
        [{"Target": "HD_29037", "Obs Window Start": "2026-11-20 17:00:00",
          "Obs Window Stop": "2026-11-21 17:00:00"}]
    ).to_csv(windows_path, index=False)

    builder = _ScienceCalendarBuilder(
        ScienceCalendarInputs(schedule_csv=schedule_path, data_dir=data_dir),
        PandoraSchedulerConfig(
            window_start=datetime(2026, 11, 20),
            window_end=datetime(2026, 11, 21),
            occultation_windows_csv=windows_path,
        ),
    )

    start = datetime(2026, 11, 20, 0, 0)
    stop = start + timedelta(hours=1)
    assert builder._occultation_target_allowed("Other_Target", start, stop)


def test_occultation_scheduler_assigns_windowed_target_only_inside_window(tmp_path):
    start = datetime(2026, 11, 20, 17, 0)
    minute = 1.0 / 1440.0
    start_mjd = Time(start).mjd
    starts = np.array([start_mjd + i * minute for i in range(3)])
    stops = starts + minute

    visibility_dir = tmp_path / "aux_targets" / "HD_29037"
    visibility_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "Time(MJD_UTC)": [start_mjd + i * minute for i in range(4)],
            "Visible": [1, 1, 1, 1],
        }
    ).to_parquet(visibility_dir / "Visibility for HD_29037.parquet", index=False)
    regular_dir = tmp_path / "aux_targets" / "Regular"
    regular_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "Time(MJD_UTC)": [start_mjd + i * minute for i in range(4)],
            "Visible": [1, 1, 1, 1],
        }
    ).to_parquet(regular_dir / "Visibility for Regular.parquet", index=False)

    occ_df = pd.DataFrame(
        {
            "Target": ["", "", ""],
            "start": ["", "", ""],
            "stop": ["", "", ""],
            "RA": [np.nan, np.nan, np.nan],
            "DEC": [np.nan, np.nan, np.nan],
        }
    )
    occ_list = pd.DataFrame([
        {"Star Name": "Regular", "RA": 1.0, "DEC": 2.0},
        {"Star Name": "HD_29037", "RA": 68.0, "DEC": 32.0},
    ])

    result, assigned = schedule_occultation_targets(
        np.array(["Regular", "HD_29037"]),
        starts,
        stops,
        start,
        start + timedelta(minutes=3),
        tmp_path / "aux_targets",
        occ_df,
        occ_list,
        "occ list",
        use_pass1=False,
        target_windows={
            "HD_29037": [(start_mjd + minute, start_mjd + 2 * minute)]
        },
    )

    assert assigned
    assert result.loc[0, "Target"] == "Regular"
    assert result.loc[1, "Target"] == "HD_29037"
    assert result.loc[2, "Target"] == "Regular"


def test_windowed_time_critical_target_is_added_to_run_manifest(tmp_path, monkeypatch):
    target_base = tmp_path / "target_definitions"
    (target_base / "time-critical").mkdir(parents=True)
    windows_path = tmp_path / "occultation_windows.csv"
    pd.DataFrame(
        [{
            "Target": "HD_29037",
            "Obs Window Start": "2026-11-20 17:00:00",
            "Obs Window Stop": "2026-11-21 17:00:00",
        }]
    ).to_csv(windows_path, index=False)
    occ_path = tmp_path / "occultation-standard_targets.csv"
    pd.DataFrame([{
        "Star Name": "Existing",
        "RA": 1.0,
        "DEC": 2.0,
        "Number of Hours Requested": 600.0,
    }]).to_csv(occ_path, index=False)

    monkeypatch.setattr(
        "pandorascheduler_rework.pipeline.rework_helper.process_target_files",
        lambda *_args, **_kwargs: pd.DataFrame([{
            "Star Name": "HD_29037",
            "RA": 68.8977,
            "DEC": 32.5798,
            "Priority": 0.0,
        }]),
    )

    _append_windowed_occultation_targets(occ_path, windows_path, target_base)
    result = pd.read_csv(occ_path)
    row = result.loc[result["Star Name"] == "HD_29037"].iloc[0]
    assert row["Number of Hours Requested"] == 24.0
    assert len(result) == 2


def test_regular_occultation_fills_window_when_windowed_target_is_not_visible(
    tmp_path, monkeypatch
):
    start = datetime(2026, 11, 20, 17, 0)
    minute = 1.0 / 1440.0
    start_mjd = Time(start).mjd
    starts = np.array([start_mjd + i * minute for i in range(3)])
    stops = starts + minute
    vis_times = np.array([start_mjd + i * minute for i in range(4)])

    monkeypatch.setattr(
        observation_utils,
        "resolve_star_visibility_file",
        lambda _base, name: Path(name),
    )
    monkeypatch.setattr(
        observation_utils,
        "load_visibility_arrays_cached",
        lambda path: (
            vis_times,
            np.zeros(len(vis_times), dtype=int)
            if str(path) == "HD_29037"
            else np.ones(len(vis_times), dtype=int),
        ),
    )

    occ_df = pd.DataFrame({
        "Target": ["", "", ""],
        "start": ["", "", ""],
        "stop": ["", "", ""],
        "RA": [np.nan, np.nan, np.nan],
        "DEC": [np.nan, np.nan, np.nan],
    })
    occ_list = pd.DataFrame([
        {"Star Name": "Regular", "RA": 1.0, "DEC": 2.0},
        {"Star Name": "HD_29037", "RA": 68.0, "DEC": 32.0},
    ])

    result, assigned = schedule_occultation_targets(
        np.array(["Regular", "HD_29037"]),
        starts,
        stops,
        start,
        start + timedelta(minutes=3),
        tmp_path,
        occ_df,
        occ_list,
        "occ list",
        use_pass1=False,
        target_windows={
            "HD_29037": [(start_mjd + minute, start_mjd + 2 * minute)]
        },
    )

    assert assigned
    assert result["Target"].tolist() == ["Regular", "Regular", "Regular"]


def test_regular_fallback_excludes_windowed_targets(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_minimal_catalogs(data_dir)

    schedule_path = tmp_path / "schedule.csv"
    pd.DataFrame(
        [{
            "Target": "Test b",
            "Observation Start": "2026-11-20 17:00:00",
            "Observation Stop": "2026-11-20 18:00:00",
        }]
    ).to_csv(schedule_path, index=False)
    windows_path = tmp_path / "occultation_windows.csv"
    pd.DataFrame(
        [{
            "Target": "HD_29037",
            "Obs Window Start": "2026-11-20 17:00:00",
            "Obs Window Stop": "2026-11-21 17:00:00",
        }]
    ).to_csv(windows_path, index=False)

    builder = _ScienceCalendarBuilder(
        ScienceCalendarInputs(schedule_csv=schedule_path, data_dir=data_dir),
        PandoraSchedulerConfig(
            window_start=datetime(2026, 11, 20),
            window_end=datetime(2026, 11, 21),
            occultation_windows_csv=windows_path,
        ),
    )
    builder.occ_catalog = pd.DataFrame(
        [
            {"Star Name": "HD_29037", "RA": 68.0, "DEC": 32.0},
            {"Star Name": "Regular", "RA": 1.0, "DEC": 2.0},
        ]
    )

    fallback = builder._select_fallback_occultation_target(
        0.0,
        0.0,
        excluded_targets={"HD_29037"},
    )

    assert fallback is not None
    assert fallback[0] == "Regular"
