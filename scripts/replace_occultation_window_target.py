#!/usr/bin/env python3
"""Replace occultation XML intervals with a visible one-off target.

This operates on an existing science calendar. It does not rerun scheduling or
visibility generation. Only sequences marked as ``occultation`` in the
provenance CSV are considered, and the replacement is made only for the part
of each sequence contained in the requested UTC window.
"""

from __future__ import annotations

import argparse
import copy
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time

from pandorascheduler_rework.xml.builder import observation_sequence


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(element: ET.Element, name: str) -> ET.Element | None:
    return next(
        (child for child in element if _local_name(child.tag) == name),
        None,
    )


def _nested_text(element: ET.Element, *names: str) -> str | None:
    current = element
    for name in names:
        current = _child(current, name)
        if current is None:
            return None
    return current.text


def _parse_time(value: object) -> datetime:
    parsed = pd.to_datetime(value, utc=True, errors="raise")
    return parsed.to_pydatetime().replace(tzinfo=None)


def _format_time(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _set_nested_text(element: ET.Element, names: tuple[str, ...], value: str) -> None:
    current = element
    for name in names:
        current = _child(current, name)
        if current is None:
            return
    current.text = value


def _qualify(element: ET.Element, namespace: str) -> None:
    """Apply the calendar namespace to an element built by the XML helper."""
    if not element.tag.startswith("{"):
        element.tag = f"{{{namespace}}}{element.tag}"
    for child in element:
        _qualify(child, namespace)


def _load_target_info(
    data_dir: Path,
    target_name: str,
    target_definition_json: Path | None = None,
) -> pd.DataFrame:
    catalog = data_dir / "occultation-standard_targets.csv"
    if not catalog.exists():
        raise FileNotFoundError(f"Occultation catalog not found: {catalog}")
    table = pd.read_csv(catalog)
    if "Star Name" not in table.columns:
        raise ValueError(f"Occultation catalog is missing 'Star Name': {catalog}")
    match = table.loc[table["Star Name"].astype(str).str.strip() == target_name]
    if match.empty:
        raise ValueError(f"Target {target_name!r} not found in {catalog}")
    result = match.iloc[[0]].copy()
    if target_definition_json is not None:
        with target_definition_json.open() as handle:
            definition = json.load(handle)
        for key in ("StarRoiDetMethod", "numPredefinedStarRois"):
            if key in definition:
                result.loc[result.index[0], key] = definition[key]
    return result


def _load_visibility(data_dir: Path, target_name: str) -> tuple[np.ndarray, np.ndarray]:
    target_dir = data_dir / "aux_targets" / target_name
    candidates = sorted(target_dir.glob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"Visibility parquet not found for {target_name}: {target_dir}"
        )
    table = pd.read_parquet(candidates[0])
    if "Time(MJD_UTC)" in table.columns:
        times = pd.to_numeric(table["Time(MJD_UTC)"], errors="coerce").to_numpy()
    elif "Time_UTC" in table.columns:
        times = Time(
            pd.to_datetime(table["Time_UTC"], utc=True).dt.to_pydatetime()
        ).mjd
    else:
        raise ValueError(f"Visibility file lacks a UTC time column: {candidates[0]}")
    visible = pd.to_numeric(table["Visible"], errors="coerce").to_numpy()
    valid = np.isfinite(times) & np.isfinite(visible)
    return times[valid], visible[valid]


def _fully_visible(
    times: np.ndarray,
    visible: np.ndarray,
    start: datetime,
    stop: datetime,
) -> bool:
    start_mjd, stop_mjd = Time([start, stop], scale="utc").mjd
    mask = (times >= start_mjd) & (times < stop_mjd)
    return bool(mask.any() and np.all(visible[mask] == 1))


def _build_replacement_sequence(
    target_info: pd.DataFrame,
    target_name: str,
    priority: object,
    start: datetime,
    stop: datetime,
    namespace: str,
) -> ET.Element:
    row = target_info.iloc[0]
    parent = ET.Element("Replacement")
    observation_sequence(
        parent,
        "000",
        target_name,
        str(priority),
        _format_time(start),
        _format_time(stop),
        float(row["RA"]),
        float(row["DEC"]),
        target_info,
    )
    replacement = parent[0]
    _qualify(replacement, namespace)
    return replacement


def _update_timing(sequence: ET.Element, start: datetime, stop: datetime) -> None:
    _set_nested_text(sequence, ("Observational_Parameters", "Timing", "Start"), _format_time(start))
    _set_nested_text(sequence, ("Observational_Parameters", "Timing", "Stop"), _format_time(stop))


def _split_record(record: dict[str, object], start: datetime, stop: datetime) -> dict[str, object]:
    result = dict(record)
    duration = (stop - start).total_seconds() / 60.0
    result["start_utc"] = _format_time(start)
    result["stop_utc"] = _format_time(stop)
    result["duration_minutes"] = duration
    try:
        fraction = float(record.get("visibility_fraction", 1.0))
    except (TypeError, ValueError):
        fraction = 1.0
    result["visible_minutes"] = duration * fraction
    result["non_visible_minutes"] = duration - result["visible_minutes"]
    return result


def replace_window(
    xml_path: Path,
    provenance_path: Path,
    output_xml: Path,
    output_provenance: Path,
    data_dir: Path,
    target_name: str,
    target_definition_json: Path | None,
    window_start: datetime,
    window_stop: datetime,
) -> int:
    provenance = pd.read_csv(provenance_path)
    required = {"visit_id", "sequence_id", "sequence_type", "start_utc", "stop_utc"}
    missing = required.difference(provenance.columns)
    if missing:
        raise ValueError(f"Provenance CSV is missing columns: {', '.join(sorted(missing))}")

    records = {
        (str(row["visit_id"]).zfill(4), str(row["sequence_id"]).zfill(3)): row.to_dict()
        for _, row in provenance.iterrows()
    }
    target_info = _load_target_info(
        data_dir,
        target_name,
        target_definition_json=target_definition_json,
    )
    visibility_times, visibility_flags = _load_visibility(data_dir, target_name)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    namespace = root.tag[1:].split("}", 1)[0] if root.tag.startswith("{") else ""
    if namespace:
        ET.register_namespace("", namespace)

    updated_records: list[dict[str, object]] = []
    replacements = 0

    for visit in [node for node in root.iter() if _local_name(node.tag) == "Visit"]:
        visit_id_node = _child(visit, "ID")
        visit_id = str(visit_id_node.text).zfill(4) if visit_id_node is not None else ""
        sequences = [child for child in list(visit) if _local_name(child.tag) == "Observation_Sequence"]
        if not sequences:
            continue

        first_index = list(visit).index(sequences[0])
        new_sequences: list[ET.Element] = []
        visit_records: list[dict[str, object]] = []

        for sequence in sequences:
            sequence_id_node = _child(sequence, "ID")
            sequence_id = (
                str(sequence_id_node.text).zfill(3)
                if sequence_id_node is not None
                else ""
            )
            record = records.get((visit_id, sequence_id))
            if record is None:
                new_sequences.append(sequence)
                continue

            start = _parse_time(record["start_utc"])
            stop = _parse_time(record["stop_utc"])
            if record.get("sequence_type") != "occultation":
                new_sequences.append(sequence)
                visit_records.append(_split_record(record, start, stop))
                continue

            overlap_start = max(start, window_start)
            overlap_stop = min(stop, window_stop)
            if overlap_stop <= overlap_start or not _fully_visible(
                visibility_times, visibility_flags, overlap_start, overlap_stop
            ):
                new_sequences.append(sequence)
                visit_records.append(_split_record(record, start, stop))
                continue

            parts: list[tuple[datetime, datetime, bool]] = []
            if start < overlap_start:
                parts.append((start, overlap_start, False))
            parts.append((overlap_start, overlap_stop, True))
            if overlap_stop < stop:
                parts.append((overlap_stop, stop, False))

            for part_start, part_stop, use_target in parts:
                if use_target:
                    replacement = _build_replacement_sequence(
                        target_info,
                        target_name,
                        record.get("priority", "0"),
                        part_start,
                        part_stop,
                        namespace,
                    )
                    new_record = _split_record(record, part_start, part_stop)
                    new_record.update(
                        {
                            "target": target_name,
                            "assignment_source": "postprocess_occultation_window",
                            "occultation_pass": "postprocess",
                            "visibility_fraction": 1.0,
                            "visible_minutes": (part_stop - part_start).total_seconds() / 60.0,
                            "non_visible_minutes": 0.0,
                        }
                    )
                    replacements += 1
                else:
                    replacement = copy.deepcopy(sequence)
                    _update_timing(replacement, part_start, part_stop)
                    new_record = _split_record(record, part_start, part_stop)
                new_sequences.append(replacement)
                visit_records.append(new_record)

        for child in sequences:
            visit.remove(child)
        for offset, sequence in enumerate(new_sequences):
            sequence_id = f"{offset + 1:03d}"
            id_node = _child(sequence, "ID")
            if id_node is not None:
                id_node.text = sequence_id
            if offset < len(visit_records):
                visit_records[offset]["visit_id"] = visit_id
                visit_records[offset]["sequence_id"] = sequence_id
            visit.insert(first_index + offset, sequence)
        updated_records.extend(visit_records)

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="\t")
    tree.write(output_xml, encoding="UTF-8", xml_declaration=True)

    output_provenance.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(updated_records, columns=provenance.columns).to_csv(
        output_provenance, index=False
    )
    return replacements


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--target-name", default="HD_29037")
    parser.add_argument("--target-definition-json", type=Path)
    parser.add_argument("--start", required=True, help="UTC start, e.g. 2026-11-20 17:00:00")
    parser.add_argument("--stop", required=True, help="UTC stop, e.g. 2026-11-21 17:00:00")
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--output-xml", type=Path)
    parser.add_argument("--output-provenance", type=Path)
    args = parser.parse_args()

    provenance = args.provenance or args.xml.with_name(
        f"{args.xml.stem}_sequence_provenance.csv"
    )
    output_xml = args.output_xml or args.xml.with_name(
        f"{args.xml.stem}_{args.target_name}_window.xml"
    )
    output_provenance = args.output_provenance or output_xml.with_name(
        f"{output_xml.stem}_sequence_provenance.csv"
    )
    replacements = replace_window(
        args.xml,
        provenance,
        output_xml,
        output_provenance,
        args.data_dir,
        args.target_name,
        args.target_definition_json,
        _parse_time(args.start),
        _parse_time(args.stop),
    )
    print(f"Replaced {replacements} occultation sequence portion(s).")
    print(f"XML: {output_xml}")
    print(f"Provenance: {output_provenance}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
