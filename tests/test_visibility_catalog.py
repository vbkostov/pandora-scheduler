from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from pandorascheduler_rework.visibility import diff as diff_module
from pandorascheduler_rework.visibility.catalog import build_visibility_catalog
from pandorascheduler_rework.visibility.config import VisibilityConfig
from pandorascheduler_rework.visibility.diff import ComparisonSummary, compare_visibility_trees
from pandorascheduler_rework.visibility.geometry import (
    build_minute_cadence,
    compute_saa_crossings,
    interpolate_gmat_ephemeris,
)


MJD_EPOCH = datetime(1858, 11, 17)


def mjd_to_datetime(mjd: float) -> datetime:
    return MJD_EPOCH + timedelta(days=mjd)


def _to_gmat_mod_julian(times: Time) -> np.ndarray:
    mjd = np.asarray(times.to_value("mjd"), dtype=float)
    return mjd - 29999.5


def test_build_visibility_catalog_generates_star_and_planet_outputs(tmp_path):
    window_start = datetime(2025, 2, 5, 0, 0, 0)
    window_end = window_start + timedelta(hours=6)

    cadence = build_minute_cadence(window_start, window_end)

    gmat_samples = Time(
        [
            window_start - timedelta(minutes=10),
            window_start,
            window_start + timedelta(hours=3),
            window_end,
            window_end + timedelta(minutes=10),
        ],
        format="datetime",
        scale="utc",
    )

    gmat_df = pd.DataFrame(
        {
            "Earth.UTCModJulian": _to_gmat_mod_julian(gmat_samples),
            "Earth.EarthMJ2000Eq.X": np.full(gmat_samples.size, -7000.0),
            "Earth.EarthMJ2000Eq.Y": np.zeros(gmat_samples.size),
            "Earth.EarthMJ2000Eq.Z": np.zeros(gmat_samples.size),
            "Pandora.EarthMJ2000Eq.X": np.zeros(gmat_samples.size),
            "Pandora.EarthMJ2000Eq.Y": np.zeros(gmat_samples.size),
            "Pandora.EarthMJ2000Eq.Z": np.zeros(gmat_samples.size),
            "Sun.EarthMJ2000Eq.X": np.zeros(gmat_samples.size),
            "Sun.EarthMJ2000Eq.Y": np.full(gmat_samples.size, 7000.0),
            "Sun.EarthMJ2000Eq.Z": np.zeros(gmat_samples.size),
            "Luna.EarthMJ2000Eq.X": np.zeros(gmat_samples.size),
            "Luna.EarthMJ2000Eq.Y": np.zeros(gmat_samples.size),
            "Luna.EarthMJ2000Eq.Z": np.full(gmat_samples.size, 7000.0),
            "Pandora.Earth.Latitude": np.full(gmat_samples.size, -20.0),
            "Pandora.Earth.Longitude": np.full(gmat_samples.size, -50.0),
        }
    )

    gmat_path = tmp_path / "gmat.csv"
    gmat_df.to_csv(gmat_path, index=False)

    period_minutes = 30.0
    period_days = period_minutes / (24.0 * 60.0)
    epoch_time = Time(window_start - timedelta(minutes=150), scale="tdb", format="datetime")
    # Ensure scalar numeric JD even if astropy returns a masked/array type
    epoch_bjd_tdb = float(np.asarray(epoch_time.jd))

    target_df = pd.DataFrame(
        {
            "Star Name": ["TestStar"],
            "Star Simbad Name": ["TestStar"],
            "Planet Name": ["TestPlanet"],
            "Planet Simbad Name": ["TestPlanet"],
            "RA": [0.0],
            "DEC": [0.0],
            "Transit Duration (hrs)": [1.0],
            "Period (days)": [period_days],
            "Transit Epoch (BJD_TDB-2400000.5)": [epoch_bjd_tdb - 2400000.5],
        }
    )
    target_path = tmp_path / "target_list.csv"
    target_df.to_csv(target_path, index=False)

    output_root = tmp_path / "visibility_outputs"

    config = VisibilityConfig(
        window_start=window_start,
        window_end=window_end,
        gmat_ephemeris=gmat_path,
        target_list=target_path,
        partner_list=None,
        output_root=output_root,
        sun_avoidance_deg=45.0,
        moon_avoidance_deg=30.0,
        earth_avoidance_deg=20.0,
        force=True,
    )

    build_visibility_catalog(config)

    star_output = output_root / "TestStar" / "Visibility for TestStar.csv"
    planet_output = output_root / "TestStar" / "TestPlanet" / "Visibility for TestPlanet.csv"

    assert star_output.exists()
    assert planet_output.exists()

    star_visibility = pd.read_csv(star_output)
    planet_visibility = pd.read_csv(planet_output)

    ephemeris = interpolate_gmat_ephemeris(gmat_path, cadence)

    star_unit_vector = np.array([1.0, 0.0, 0.0])

    def _separation_deg(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1)
        dot = vectors @ star_unit_vector
        cos_theta = np.clip(dot / norms, -1.0, 1.0)
        return np.round(np.degrees(np.arccos(cos_theta)), 3)

    earth_sep = _separation_deg(ephemeris.earth_pc)
    sun_sep = _separation_deg(ephemeris.sun_pc)
    moon_sep = _separation_deg(ephemeris.moon_pc)

    visible = (sun_sep > 45.0) & (moon_sep > 30.0) & (earth_sep > 20.0)
    visible = np.round(visible.astype(float), 1)

    saa_crossing = np.round(
        compute_saa_crossings(
            ephemeris.spacecraft_lat_deg, ephemeris.spacecraft_lon_deg
        ),
        1,
    )

    expected_star = pd.DataFrame(
        {
            "Time(MJD_UTC)": np.round(cadence.mjd_utc, 6),
            "SAA_Crossing": saa_crossing,
            "Visible": visible,
            "Earth_Sep": earth_sep,
            "Moon_Sep": moon_sep,
            "Sun_Sep": sun_sep,
        }
    )

    # Compare columns and numeric values with tolerance rather than strict equality
    assert list(star_visibility.columns) == list(expected_star.columns)
    for col in expected_star.columns:
        exp_vals = np.asarray(expected_star[col], dtype=float)
        got_vals = np.asarray(star_visibility[col].to_numpy(), dtype=float)
        assert np.allclose(got_vals, exp_vals, atol=1e-6, equal_nan=True)

    assert not planet_visibility.empty
    np.testing.assert_array_equal(
        planet_visibility["Transits"].to_numpy(),
        np.arange(len(planet_visibility)),
    )

    star_time_mjd = star_visibility["Time(MJD_UTC)"].to_numpy(dtype=float)
    star_visible = star_visibility["Visible"].to_numpy()
    star_saa = star_visibility["SAA_Crossing"].to_numpy()

    star_time = Time(star_time_mjd, format="mjd", scale="utc")
    star_time_iso = Time(star_time.iso, format="iso", scale="utc")
    # Normalize to python datetimes to avoid masked/np.datetime64 issues
    raw_datetimes = star_time_iso.to_value("datetime")
    star_datetimes = [pd.to_datetime(dt).to_pydatetime() for dt in np.asarray(raw_datetimes)]

    visible_times = {
        dt for dt, flag in zip(star_datetimes, star_visible) if flag == 1.0
    }
    saa_times = {
        dt for dt, flag in zip(star_datetimes, star_saa) if flag == 1.0
    }

    assert "SAA_Overlap" in planet_visibility.columns
    assert "Transit_Overlap" not in planet_visibility.columns

    for _, row in planet_visibility.iterrows():
        start_mjd = float(row["Transit_Start"])
        end_mjd = float(row["Transit_Stop"])
        start_dt = mjd_to_datetime(start_mjd).replace(second=0, microsecond=0)
        end_dt = mjd_to_datetime(end_mjd).replace(second=0, microsecond=0)
        minute_range = pd.date_range(start_dt, end_dt, freq="min").to_pydatetime()
        if len(minute_range) == 0:
            assert row["Transit_Coverage"] == 0.0
            continue
        overlap = visible_times.intersection(minute_range)
        expected_cov = len(overlap) / len(minute_range)
        assert np.isclose(row["Transit_Coverage"], expected_cov)

        saa_overlap = saa_times.intersection(minute_range)
        expected_saa = len(saa_overlap) / len(minute_range)
        assert np.isclose(row["SAA_Overlap"], expected_saa)


def test_resolve_star_coord_uses_catalog_coordinates():
    """Test that _resolve_star_coord uses catalog coordinates (no Simbad)."""
    from pandorascheduler_rework.visibility import catalog

    row = pd.Series(
        {
            "Star Name": "Foo",
            "RA": 123.4,
            "DEC": 56.7,
        }
    )
    metadata = {"Foo": (123.4, 56.7)}

    coord = catalog._resolve_star_coord(row, metadata)

    assert np.isclose(coord.ra.deg, 123.4)
    assert np.isclose(coord.dec.deg, 56.7)



def test_resolve_star_coord_uses_metadata_fallback():
    """Test that _resolve_star_coord uses metadata as fallback."""
    from pandorascheduler_rework.visibility import catalog

    row = pd.Series(
        {
            "Star Name": "Bar",
            "RA": np.nan,  # Missing in row
            "DEC": np.nan,  # Missing in row
        }
    )
    metadata = {"Bar": (10.0, -11.0)}

    coord = catalog._resolve_star_coord(row, metadata)

    assert coord.ra.deg == 10.0
    assert coord.dec.deg == -11.0



def test_resolve_star_coord_raises_error_when_missing():
    """Test that _resolve_star_coord raises error when coordinates missing."""
    from pandorascheduler_rework.visibility import catalog
    import pytest

    row = pd.Series(
        {
            "Star Name": "Baz",
            "RA": np.nan,
            "DEC": np.nan,
        }
    )
    metadata = {}  # No fallback

    with pytest.raises(RuntimeError, match="No coordinates found in catalog for Baz"):
        catalog._resolve_star_coord(row, metadata)


def _write_csv(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_compare_visibility_trees_detects_mismatches(tmp_path):
    legacy_root = tmp_path / "legacy"
    rework_root = tmp_path / "rework"

    legacy_star = legacy_root / "Star" / "Visibility for Star.csv"
    rework_star = rework_root / "Star" / "Visibility for Star.csv"

    legacy_planet = legacy_root / "Star" / "Planet" / "Visibility for Planet.csv"
    rework_planet = rework_root / "Star" / "Planet" / "Visibility for Planet.csv"

    _write_csv(
        legacy_star,
        "Time(MJD_UTC),SAA_Crossing,Visible,Earth_Sep\n"
        "1,0,1,90\n"
        "2,1,0,91\n",
    )
    _write_csv(
        rework_star,
        "Time(MJD_UTC),SAA_Crossing,Visible,Earth_Sep\n"
        "1,0,1,90.001\n"
        "2,1,1,91.002\n",
    )

    _write_csv(
        legacy_planet,
        "Transits,Transit_Start,Transit_Stop,Transit_Coverage,SAA_Overlap\n"
        "0,1,2,0.5,0.0\n",
    )
    _write_csv(
        rework_planet,
        "Transits,Transit_Start,Transit_Stop,Transit_Coverage,Transit_Overlap,SAA_Overlap\n"
        "0,1,2,0.6,0.0,0.1\n",
    )

    summary = compare_visibility_trees(legacy_root, rework_root, atol=1e-4)

    assert isinstance(summary, ComparisonSummary)
    assert len(summary.missing_in_legacy) == 0
    assert len(summary.missing_in_rework) == 0

    differing = {result.path: result for result in summary.differing_files}
    assert Path("Star/Visibility for Star.csv") in differing
    assert differing[Path("Star/Visibility for Star.csv")].visible_mismatches == 1
    assert differing[Path("Star/Visibility for Star.csv")].numeric_deltas["Earth_Sep"] == pytest.approx(0.002)

    assert Path("Star/Planet/Visibility for Planet.csv") in differing
    planet_result = differing[Path("Star/Planet/Visibility for Planet.csv")]
    assert planet_result.extra_columns_rework == ("Transit_Overlap",)
    assert planet_result.numeric_deltas["Transit_Coverage"] == pytest.approx(0.1)
    assert planet_result.numeric_deltas["SAA_Overlap"] == pytest.approx(0.1)


def test_compare_and_print_returns_nonzero_for_differences(tmp_path, monkeypatch):
    legacy_root = tmp_path / "legacy"
    rework_root = tmp_path / "rework"

    _write_csv(
        legacy_root / "file.csv",
        "Visible\n0\n",
    )
    _write_csv(
        rework_root / "file.csv",
        "Visible\n1\n",
    )

    captured = []

    def fake_print(message: str) -> None:
        captured.append(message)

    monkeypatch.setattr(diff_module, "print", fake_print, raising=False)

    summary = diff_module.compare_and_print(legacy_root, rework_root)

    assert summary.identical is False
    assert any("Total differing files: 1" in line for line in captured)


def test_compare_and_print_returns_zero_for_matches(tmp_path, monkeypatch):
    legacy_root = tmp_path / "legacy"
    rework_root = tmp_path / "rework"

    _write_csv(
        legacy_root / "file.csv",
        "Visible\n1\n",
    )
    _write_csv(
        rework_root / "file.csv",
        "Visible\n1\n",
    )

    captured = []

    def fake_print(message: str) -> None:
        captured.append(message)

    monkeypatch.setattr(diff_module, "print", fake_print, raising=False)

    summary = diff_module.compare_and_print(legacy_root, rework_root)

    assert summary.identical is True
    assert any("Total differing files: 0" in line for line in captured)
