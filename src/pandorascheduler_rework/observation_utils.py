"""Utilities for scheduling observations and building observation sequences.

This module provides functions for:
- Creating observation sequence XML elements
- Processing visibility windows
- Managing target parameters and observational constraints
- Handling NIRDA and VDA payload parameters
- Building and manipulating observation schedules

This is a refactored version of the legacy helper_codes module with improved
naming and documentation while maintaining compatibility with the existing
scheduling pipeline.
"""

from __future__ import annotations

import ast
from datetime import datetime, timedelta
import functools
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, cast

import re
import logging

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from astropy.time import Time
from tqdm import tqdm

from pandorascheduler_rework.targets.manifest import build_target_manifest


LOGGER = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parent
LEGACY_PACKAGE_DIR = PACKAGE_DIR.parent / "pandorascheduler"
DATA_ROOTS = [PACKAGE_DIR / "data", LEGACY_PACKAGE_DIR / "data"]
_PLACEHOLDER_MARKERS = {"SET_BY_TARGET_DEFINITION_FILE", "SET_BY_SCHEDULER"}


def general_parameters(
    obs_sequence_duration: int = 90,
    occ_sequence_limit: int = 30,
) -> Tuple[int, int]:
    return obs_sequence_duration, occ_sequence_limit


def observation_sequence(
    visit,
    obs_seq_id: str,
    target_name: str,
    priority: str,
    start,
    stop,
    ra,
    dec,
    targ_info: pd.DataFrame,
):
    sequence_element = ET.SubElement(visit, "Observation_Sequence")
    ET.SubElement(sequence_element, "ID").text = obs_seq_id

    observational_parameters = _observational_parameters(
        target_name, priority, start, stop, ra, dec
    )

    obs_parameters = ET.SubElement(sequence_element, "Observational_Parameters")
    for key, value in observational_parameters.items():
        obs_param_element = ET.SubElement(obs_parameters, key)
        if key in {"Timing", "Boresight"}:
            for index in range(2):
                sub_element = ET.SubElement(obs_param_element, value[index])
                sub_element.text = value[index + 2]
        else:
            obs_param_element.text = str(value)

    diff_in_seconds = _duration_in_seconds(start, stop)

    payload_parameters = ET.SubElement(sequence_element, "Payload_Parameters")
    _populate_nirda_parameters(payload_parameters, targ_info, diff_in_seconds)
    _populate_vda_parameters(payload_parameters, targ_info, diff_in_seconds)

    return sequence_element


def remove_short_sequences(array, sequence_too_short: int):
    cleaned = np.asarray(array, dtype=float).copy()
    start_index = None
    spans: List[Tuple[int, int]] = []

    for idx, value in enumerate(cleaned):
        if value == 1 and start_index is None:
            start_index = idx
            continue
        if value == 0 and start_index is not None:
            if idx - start_index < sequence_too_short:
                spans.append((start_index, idx - 1))
            start_index = None

    if start_index is not None and len(cleaned) - start_index < sequence_too_short:
        spans.append((start_index, len(cleaned) - 1))

    for start_idx, stop_idx in spans:
        cleaned[start_idx : stop_idx + 1] = 0.0

    return cleaned, spans


def break_long_sequences(start, end, step: timedelta):
    ranges: List[Tuple[datetime, datetime]] = []
    current = start
    while current < end:
        next_val = min(current + step, end)
        ranges.append((current, next_val))
        current = next_val
    return ranges


def _observational_parameters(target_name, priority, start, stop, ra, dec):
    try:
        start_format = datetime.strftime(start, "%Y-%m-%dT%H:%M:%SZ")
        stop_format = datetime.strftime(stop, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError, AttributeError):
        start_format, stop_format = start, stop

    try:
        ra_value = f"{float(ra)}"
        dec_value = f"{float(dec)}"
    except (TypeError, ValueError):
        ra_value, dec_value = "-999.0", "-999.0"

    return {
        "Target": target_name,
        "Priority": f"{priority}",
        "Timing": ["Start", "Stop", start_format, stop_format],
        "Boresight": ["RA", "DEC", ra_value, dec_value],
    }


def _duration_in_seconds(start, stop) -> float:
    if isinstance(stop, datetime) and isinstance(start, datetime):
        return (stop - start).total_seconds()

    if isinstance(stop, str) and isinstance(start, str):
        try:
            stop_dt = datetime.strptime(stop, "%Y-%m-%dT%H:%M:%SZ")
            start_dt = datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ")
            return (stop_dt - start_dt).total_seconds()
        except (ValueError, TypeError):
            return 0.0

    return 0.0


def _populate_nirda_parameters(payload_parameters, targ_info: pd.DataFrame, diff_in_seconds: float) -> None:
    if targ_info.empty:
        ET.SubElement(payload_parameters, "AcquireInfCamImages")
        return

    nirda_columns = targ_info.columns[targ_info.columns.str.startswith("NIRDA_")]
    nirda_element = ET.SubElement(payload_parameters, "AcquireInfCamImages")

    if nirda_columns.empty:
        return

    columns_to_ignore = {
        "IncludeFieldSolnsInResp",
        "NIRDA_TargetID",
        "NIRDA_SC_Integrations",
        "NIRDA_FramesPerIntegration",
        "NIRDA_IntegrationTime_s",
    }

    row = targ_info.iloc[0]

    for nirda_key, nirda_value in row[nirda_columns].items():
        column_name = str(nirda_key)
        if pd.isna(nirda_value):
            continue

        if column_name not in columns_to_ignore:
            ET.SubElement(nirda_element, column_name.replace("NIRDA_", "")).text = str(nirda_value)
            continue

        if column_name == "NIRDA_TargetID":
            ET.SubElement(nirda_element, "TargetID").text = _target_identifier(row)
            continue

        if column_name == "NIRDA_SC_Integrations":
            integration_time = row.get("NIRDA_IntegrationTime_s")
            if pd.notna(integration_time) and integration_time:
                integrations = int(np.round(diff_in_seconds / integration_time))
                integrations = max(integrations, 0)
                ET.SubElement(nirda_element, "SC_Integrations").text = str(integrations)
            continue


def _populate_vda_parameters(payload_parameters, targ_info: pd.DataFrame, diff_in_seconds: float) -> None:
    if targ_info.empty:
        ET.SubElement(payload_parameters, "AcquireVisCamScienceData")
        return

    vda_columns = targ_info.columns[targ_info.columns.str.startswith("VDA_")]
    vda_element = ET.SubElement(payload_parameters, "AcquireVisCamScienceData")

    if vda_columns.empty:
        return

    row = targ_info.iloc[0]
    columns_to_ignore = {
        "VDA_NumExposuresMax",
        "VDA_NumTotalFramesRequested",
        "VDA_TargetID",
        "VDA_TargetRA",
        "VDA_TargetDEC",
        "VDA_StarRoiDetMethod",
        "VDA_numPredefinedStarRois",
        "VDA_PredefinedStarRoiRa",
        "VDA_PredefinedStarRoiDec",
        "VDA_IntegrationTime_s",
        "VDA_MaxNumStarRois",
    }

    for vda_key, vda_value in row[vda_columns].items():
        column_name = str(vda_key)
        if pd.isna(vda_value):
            continue

        shortened_key = column_name.replace("VDA_", "")

        if column_name not in columns_to_ignore:
            ET.SubElement(vda_element, shortened_key).text = str(vda_value)
            continue

        if column_name == "VDA_TargetID":
            ET.SubElement(vda_element, "TargetID").text = _target_identifier(row)
            continue

        if column_name == "VDA_TargetRA":
            ET.SubElement(vda_element, "TargetRA").text = str(row.get("RA", vda_value))
            continue

        if column_name == "VDA_TargetDEC":
            ET.SubElement(vda_element, "TargetDEC").text = str(row.get("DEC", vda_value))
            continue

        if column_name == "VDA_StarRoiDetMethod":
            value = row.at[column_name]
            fallback = row.get("StarRoiDetMethod") if isinstance(value, str) and value in _PLACEHOLDER_MARKERS else value

            if fallback is None:
                continue
            if isinstance(fallback, str) and fallback in _PLACEHOLDER_MARKERS:
                continue
            if isinstance(fallback, float) and pd.isna(fallback):
                continue

            try:
                fallback_value = int(fallback)
            except (TypeError, ValueError):
                fallback_value = fallback

            ET.SubElement(vda_element, "StarRoiDetMethod").text = str(fallback_value)
            continue

        if column_name == "VDA_MaxNumStarRois":
            method = row.get("StarRoiDetMethod")
            if method == 1:
                value = 0
            elif method == 2:
                value = 9
            else:
                value = vda_value
            ET.SubElement(vda_element, "MaxNumStarRois").text = str(int(value))
            continue

        if column_name == "VDA_numPredefinedStarRois":
            method = row.get("StarRoiDetMethod")
            if method == 2:
                continue
            field = row.get("numPredefinedStarRois")
            text_value = str(field) if pd.notna(field) else "-9999"
            ET.SubElement(vda_element, "numPredefinedStarRois").text = text_value
            continue

        if column_name in {"VDA_PredefinedStarRoiRa", "VDA_PredefinedStarRoiDec"}:
            method = row.get("StarRoiDetMethod")
            if method == 2:
                continue
            roi_coord_columns = [
                col
                for col in targ_info.columns
                if col.startswith("ROI_coord_") and col != "ROI_coord_epoch"
            ]
            roi_coord_values = targ_info[roi_coord_columns].dropna(axis=1)
            if roi_coord_values.empty:
                continue
            try:
                coordinates = np.asarray(
                    [ast.literal_eval(item) for item in roi_coord_values.iloc[0]]
                )
            except (ValueError, SyntaxError):
                continue

            element = ET.SubElement(vda_element, shortened_key)
            for index, coordinate in enumerate(coordinates):
                tag = "RA" if column_name == "VDA_PredefinedStarRoiRa" else "Dec"
                sub = ET.SubElement(element, f"{tag}{index + 1}")
                sub.text = f"{coordinate[0 if tag == 'RA' else 1]:.6f}"
            continue

        if column_name == "VDA_NumTotalFramesRequested":
            exposure_time_us = row.get("VDA_ExposureTime_us")
            frames_per_coadd = row.get("VDA_FramesPerCoadd")
            if pd.notna(exposure_time_us) and pd.notna(frames_per_coadd) and frames_per_coadd:
                exposure_seconds = 1e-6 * float(exposure_time_us)
                if exposure_seconds > 0:
                    coadd = int(frames_per_coadd)
                    frames = int(np.floor(diff_in_seconds / exposure_seconds / coadd) * coadd)
                    ET.SubElement(vda_element, "NumTotalFramesRequested").text = str(max(frames, 0))
            continue


def _target_identifier(row: pd.Series) -> str:
    planet = row.get("Planet Name") if isinstance(row, pd.Series) else None
    if planet is not None and pd.notna(planet):
        return re.sub(r"\s+([A-Za-z])$", r"\1", str(planet))

    star = row.get("Star Name") if isinstance(row, pd.Series) else None
    if star is not None and pd.notna(star):
        return str(star)

    return ""


def schedule_occultation_targets(
    v_names,
    starts,
    stops,
    visit_start,
    visit_stop,
    path,
    o_df,
    o_list,
    try_occ_targets: str,
):
    starts_array = np.asarray(starts, dtype=float)
    stops_array = np.asarray(stops, dtype=float)

    schedule = pd.DataFrame(
        {
            "Stop": stops_array,
            "Target": pd.Series([None] * len(starts_array), dtype=object),
            "Visibility": np.nan,
        },
        index=pd.Index(starts_array, name="Start"),
    )

    if "Target" in o_df.columns:
        o_df["Target"] = o_df["Target"].astype(object)

    if "Visibility" not in o_df.columns:
        o_df["Visibility"] = np.nan

    description = (
        f"{visit_start} to {visit_stop}: Searching for occultation target from {try_occ_targets}"
        if visit_start is not None and visit_stop is not None
        else f"Searching for occultation target from {try_occ_targets}"
    )

    base_path = Path(path) if path is not None else None

    for v_name in tqdm(v_names, desc=description, leave=False):
        file_path = _resolve_visibility_file(v_name, base_path)
        if file_path is None:
            LOGGER.debug("Skipping %s; visibility file not found", v_name)
            continue

        vis_times, visibility = _load_visibility_data(file_path)

        for idx, (start, stop) in enumerate(zip(starts_array, stops_array)):
            if pd.isna(schedule.loc[start, "Target"]):
                interval_mask = (vis_times >= start) & (vis_times <= stop)

                if np.all(visibility[interval_mask] == 1):
                    schedule.loc[start, "Target"] = v_name
                    schedule.loc[start, "Visibility"] = 1

                    match = o_list.loc[o_list["Star Name"] == v_name]
                    if match.empty:
                        continue
                    match_row = match.iloc[0]
                    o_df.loc[idx, "Target"] = v_name
                    o_df.loc[idx, "RA"] = match_row["RA"]
                    o_df.loc[idx, "DEC"] = match_row["DEC"]
                    o_df.loc[idx, "Visibility"] = 1
                else:
                    if pd.isna(schedule.loc[start, "Visibility"]):
                        schedule.loc[start, "Visibility"] = 0
                        o_df.loc[idx, "Visibility"] = 0

        if not schedule["Target"].isna().any():
            return o_df, True

    mask = schedule["Target"].isna()
    schedule.loc[mask, "Target"] = "No target"
    schedule.loc[mask, "Visibility"] = 0

    o_df.loc[o_df["Target"].isna(), "Target"] = "No target"
    o_df.loc[o_df["Visibility"].isna(), "Visibility"] = 0

    return o_df, False


def save_observation_time_report(
    all_target_obs_time: Dict[str, timedelta],
    target_list: pd.DataFrame,
    output_path,
):
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    primary_targets = {
        str(name)
        for name in target_list.get("Planet Name", pd.Series(dtype=object)).dropna()
    }

    with output_file.open("w", encoding="utf-8") as handle:
        handle.write("Target,Is Primary,Total Observation Time (hours)\n")
        for target, duration in all_target_obs_time.items():
            label = str(target)
            is_primary = "Yes" if label in primary_targets else "No"
            hours = duration.total_seconds() / 3600
            handle.write(f"{label},{is_primary},{hours:.2f}\n")

    return output_file


def check_if_transits_in_obs_window(
    tracker: pd.DataFrame,
    temp_df: pd.DataFrame,
    target_list: pd.DataFrame,
    start: datetime,
    pandora_start: datetime,
    pandora_stop: datetime,
    sched_start: datetime,
    sched_stop: datetime,
    obs_rng: pd.DatetimeIndex,
    obs_window: timedelta,
    sched_wts: Sequence[float],
    transit_coverage_min: float,
):

    result_df = temp_df.copy()
    columns = [
        "Planet Name",
        "RA",
        "DEC",
        "Obs Start",
        "Obs Gap Time",
        "Transit Coverage",
        "SAA Overlap",
        "Schedule Factor",
        "Transit Factor",
        "Quality Factor",
        "Comments",
    ]

    planet_lookup = target_list.set_index("Planet Name")

    for i, row in tracker.iterrows():
        planet_name = row.get("Planet Name")
        if pd.isna(planet_name):
            continue

        mask = tracker["Planet Name"] == planet_name
        needed_series = cast(pd.Series, tracker.loc[mask, "Transits Needed"])
        if needed_series.empty:
            continue
        transits_needed = float(needed_series.iloc[0])

        if transits_needed == 0:
            continue

        if planet_name not in planet_lookup.index:
            LOGGER.debug("Planet %s missing from target list; skipping", planet_name)
            continue

        star_name = str(planet_lookup.loc[planet_name, "Star Name"])
        visibility_file = _planet_visibility_file(star_name, str(planet_name))
        if visibility_file is None:
            LOGGER.debug(
                "Skipping %s; visibility file not found under %s",
                planet_name,
                star_name,
            )
            continue

        planet_data = pd.read_csv(visibility_file)
        if planet_data.empty:
            continue

        planet_data = planet_data.drop(
            planet_data.index[
                planet_data["Transit_Coverage"] < transit_coverage_min
            ]
        ).reset_index(drop=True)
        if planet_data.empty:
            continue

        start_times = Time(
            planet_data["Transit_Start"].to_numpy(), format="mjd", scale="utc"
        ).to_value("datetime")
        stop_times = Time(
            planet_data["Transit_Stop"].to_numpy(), format="mjd", scale="utc"
        ).to_value("datetime")

        start_series = pd.Series(list(cast(Sequence[datetime], start_times)))
        stop_series = pd.Series(list(cast(Sequence[datetime], stop_times)))

        valid_mask = start_series >= start
        start_series = start_series.loc[valid_mask].reset_index(drop=True)
        stop_series = stop_series.loc[valid_mask].reset_index(drop=True)
        planet_data = planet_data.loc[valid_mask].reset_index(drop=True)

        if start_series.empty:
            continue

        lifetime_mask = (pandora_start <= start_series) & (stop_series <= pandora_stop)
        schedule_mask = (sched_start <= start_series) & (stop_series <= sched_stop)
        lifetime_count = int(lifetime_mask.sum())
        schedule_count = int(schedule_mask.sum())

        tracker.loc[mask, "Transits Left in Lifetime"] = lifetime_count
        tracker.loc[mask, "Transits Left in Schedule"] = schedule_count
        tracker.loc[mask, "Transit Priority"] = lifetime_count - transits_needed

        start_series = start_series.dt.floor("min")
        stop_series = stop_series.dt.floor("min")

        early_start = stop_series - timedelta(hours=20)
        late_start = start_series - timedelta(hours=4)

        try:
            ra_tar = float(row["RA"])
            dec_tar = float(row["DEC"])
        except (TypeError, ValueError):
            ra_tar = row.get("RA")
            dec_tar = row.get("DEC")

        transits_left = float(lifetime_count)
        if transits_needed == 0:
            continue

        coverage_values = planet_data["Transit_Coverage"].to_numpy(dtype=float, copy=False)
        saa_values = planet_data["SAA_Overlap"].to_numpy(dtype=float, copy=False)

        for j, (window_start, window_stop) in enumerate(zip(early_start, late_start)):
            if window_start > window_stop:
                continue

            start_range = pd.date_range(window_start, window_stop, freq="min")
            overlap_times = obs_rng.intersection(start_range)
            if overlap_times.empty:
                continue

            obs_start = overlap_times[0]
            gap_time = obs_start - obs_rng[0]
            schedule_factor = 1 - (gap_time / obs_window)
            transit_coverage = float(coverage_values[j])
            saa_overlap = float(saa_values[j])
            quality_factor = (
                (sched_wts[0] * transit_coverage)
                + (sched_wts[1] * (1 - saa_overlap))
                + (sched_wts[2] * schedule_factor)
            )

            candidate = pd.DataFrame(
                [
                    [
                        planet_name,
                        ra_tar,
                        dec_tar,
                        obs_start,
                        gap_time,
                        transit_coverage,
                        saa_overlap,
                        schedule_factor,
                        transits_left / transits_needed,
                        quality_factor,
                        np.nan,
                    ]
                ],
                columns=columns,
            )

            if result_df.empty:
                result_df = candidate
            else:
                result_df = pd.concat([result_df, candidate], ignore_index=True)

    return result_df


def _default_target_definition_base() -> Path:
    env_value = os.environ.get("PANDORA_TARGET_DEFINITION_BASE")
    if env_value:
        candidate = Path(env_value).expanduser()
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(
            f"Configured target definition base not found: {candidate}"
        )

    fallback = (
        Path(__file__).resolve().parents[3]
        / "comparison_outputs"
        / "target_definition_files_limited"
    )
    if fallback.is_dir():
        return fallback

    raise FileNotFoundError(
        "Unable to locate target definition files. Set the "
        "PANDORA_TARGET_DEFINITION_BASE environment variable to the root of "
        "the PandoraTargetList repository."
    )


def process_target_files(keyword: str, *, base_path: Path | None = None):
    """Return the scheduler manifest for *keyword* targets.

    Parameters
    ----------
    keyword
        Category name matching a directory in the target definition
        repository, e.g. ``"exoplanet"``.
    base_path
        Optional override for the target definition repository root. When
        omitted the function first honours the
        ``PANDORA_TARGET_DEFINITION_BASE`` environment variable and finally
        falls back to the curated fixture subset under
        ``comparison_outputs/target_definition_files_limited``.
    """

    base_dir = Path(base_path) if base_path is not None else _default_target_definition_base()
    return build_target_manifest(keyword, base_dir)


def create_aux_list(target_definition_files: Sequence[str], package_dir):
    """Build ``aux_list_new.csv`` from the provided target manifest CSVs.

    The legacy helper concatenated the partner target lists that share a common
    column set.  Reimplement the same behaviour explicitly to avoid importing
    the legacy module.
    """

    if not target_definition_files:
        raise ValueError("target_definition_files must contain at least one entry")

    data_dir = Path(package_dir) / "data"
    csv_paths: List[Path] = []
    for name in target_definition_files:
        candidate = data_dir / f"{name}_targets.csv"
        if candidate.exists():
            csv_paths.append(candidate)
        else:
            LOGGER.warning("Aux list source missing: %s", candidate)

    if not csv_paths:
        raise FileNotFoundError(
            "No target definition CSVs found; unable to build aux_list_new.csv"
        )

    dataframes = [pd.read_csv(path) for path in csv_paths]

    common_columns = set(dataframes[0].columns)
    for frame in dataframes[1:]:
        common_columns &= set(frame.columns)

    if not common_columns:
        raise ValueError("Target lists share no common columns; cannot build aux list")

    primary_column_order = [col for col in dataframes[0].columns if col in common_columns]

    trimmed = [frame.loc[:, primary_column_order] for frame in dataframes]
    combined = pd.concat(trimmed, ignore_index=True).drop_duplicates().reset_index(drop=True)

    output_path = data_dir / "aux_list_new.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    return output_path


# Pre-compile regex pattern for performance
_PLANET_SUFFIX_PATTERN = re.compile(r"\s+[a-z]$", flags=re.ASCII)


def build_visibility_path(base_dir: Path, star_name: str, target_name: str) -> Path:
    """Build consistent visibility file path.
    
    Centralizes the pattern: base_dir / star_name / target_name / "Visibility for {target_name}.csv"
    Used throughout scheduler to avoid repeated f-string formatting.
    """
    return base_dir / star_name / target_name / f"Visibility for {target_name}.csv"


def build_star_visibility_path(base_dir: Path, star_name: str) -> Path:
    """Build visibility path for a star (no planet subdirectory).
    
    Pattern: base_dir / star_name / "Visibility for {star_name}.csv"
    """
    return base_dir / star_name / f"Visibility for {star_name}.csv"


def remove_suffix(value: str) -> str:
    """Return *value* with a trailing " planet suffix" removed.

    The legacy helper accepted identifiers like ``"WASP-107 b"`` and stripped the
    final ``" b"`` so that callers could recover the stellar host name.  The
    `rework` pipeline relies on the same behaviour when looking up per-star
    visibility files.  Reimplement the logic locally to avoid importing the
    legacy module for such a small utility.
    """

    if not value:
        return value

    return _PLANET_SUFFIX_PATTERN.sub("", value)


@functools.lru_cache(maxsize=32)
def _load_visibility_data(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and cache visibility data from CSV file.
    
    This function is cached to avoid repeatedly reading the same file
    when scheduling multiple observation windows for the same target.
    Cache size is limited to 32 files (~256 MB memory) to balance
    performance with memory usage.
    
    Parameters
    ----------
    file_path
        Path to the visibility CSV file.
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Time array (MJD_UTC) and visibility flags.
    """
    vis = pd.read_csv(file_path, usecols=["Time(MJD_UTC)", "Visible"])
    return vis["Time(MJD_UTC)"].to_numpy(), vis["Visible"].to_numpy()


def _resolve_visibility_file(target_name: str, base_path: Path | None) -> Path | None:
    candidates: list[Path] = []

    for data_root in DATA_ROOTS:
        for subdir in ("targets", "aux_targets"):
            candidates.append(
                build_star_visibility_path(data_root / subdir, target_name)
            )

    if base_path is not None:
        candidate = build_star_visibility_path(base_path, target_name)
        candidates.append(candidate)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


def _planet_visibility_file(star_name: str, planet_name: str) -> Path | None:
    for data_root in DATA_ROOTS:
        candidate = build_visibility_path(
            data_root / "targets", star_name, planet_name
        )
        if candidate.is_file():
            return candidate
    return None
