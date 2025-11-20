#!/usr/bin/env python3
"""Inspect differences between legacy and rework science calendar XML files."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple
import xml.etree.ElementTree as ET

NAMESPACE = {"p": "/pandora/calendar/"}


@dataclass(frozen=True)
class SequenceInfo:
    """Simplified representation of a single observation sequence."""

    visit_id: str
    seq_id: str
    target: str
    start: str
    stop: str
    priority: str


@dataclass(frozen=True)
class CalendarData:
    """Parsed science calendar contents."""

    sequences: List[SequenceInfo]
    visits: List[str]


def parse_arguments(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Pandora legacy vs rework science calendar XML outputs",
    )
    parser.add_argument("legacy", type=Path, help="Path to the legacy XML file")
    parser.add_argument("rework", type=Path, help="Path to the rework XML file")
    parser.add_argument(
        "--target",
        help="Display detailed sequence information for the specified target",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of mismatched targets to display (default: 20)",
    )
    parser.add_argument(
        "--include-equal",
        action="store_true",
        help="Show targets with matching sequence counts as well",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def load_calendar(path: Path) -> CalendarData:
    if not path.is_file():
        raise FileNotFoundError(f"Science calendar XML not found: {path}")

    tree = ET.parse(path)
    root = tree.getroot()

    sequences: List[SequenceInfo] = []
    visits: List[str] = []

    for visit in root.findall("p:Visit", NAMESPACE):
        visit_id = _clean_text(visit.findtext("p:ID", default="", namespaces=NAMESPACE))
        if not visit_id:
            visit_id = f"visit-{len(visits) + 1:04d}"
        visits.append(visit_id)

        for seq in visit.findall("p:Observation_Sequence", NAMESPACE):
            seq_id = _clean_text(seq.findtext("p:ID", default="", namespaces=NAMESPACE))
            target = _clean_text(
                seq.findtext("p:Observational_Parameters/p:Target", default="", namespaces=NAMESPACE)
            )
            start = _clean_text(
                seq.findtext("p:Observational_Parameters/p:Timing/p:Start", default="", namespaces=NAMESPACE)
            )
            stop = _clean_text(
                seq.findtext("p:Observational_Parameters/p:Timing/p:Stop", default="", namespaces=NAMESPACE)
            )
            priority = _clean_text(
                seq.findtext("p:Observational_Parameters/p:Priority", default="", namespaces=NAMESPACE)
            )
            sequences.append(
                SequenceInfo(
                    visit_id=visit_id,
                    seq_id=seq_id,
                    target=target,
                    start=start,
                    stop=stop,
                    priority=priority,
                )
            )

    return CalendarData(sequences=sequences, visits=visits)


def _clean_text(value: str) -> str:
    return value.strip() if value else ""


def summarise_counts(label: str, data: CalendarData) -> Counter:
    counter = Counter(seq.target for seq in data.sequences if seq.target)
    print(f"{label}: {len(data.visits)} visits, {len(data.sequences)} sequences, {len(counter)} targets")
    return counter


def compute_differences(
    legacy_counts: Counter,
    rework_counts: Counter,
) -> List[Tuple[str, int, int, int]]:
    results: List[Tuple[str, int, int, int]] = []
    for target in sorted(set(legacy_counts) | set(rework_counts)):
        legacy_total = legacy_counts.get(target, 0)
        rework_total = rework_counts.get(target, 0)
        diff = rework_total - legacy_total
        results.append((target, legacy_total, rework_total, diff))
    results.sort(key=lambda item: (abs(item[3]), item[0]), reverse=True)
    return results


def show_mismatches(
    differences: List[Tuple[str, int, int, int]],
    top: int,
    include_equal: bool,
) -> None:
    rows = differences if include_equal else [row for row in differences if row[3] != 0]
    if not rows:
        print("No mismatched targets found.")
        return

    limit = top if top > 0 else len(rows)
    print("Target sequence differences (rework minus legacy):")
    for target, legacy_total, rework_total, diff in rows[:limit]:
        if diff > 0:
            status = "extra in rework"
        elif diff < 0:
            status = "missing from rework"
        else:
            status = "matching"
        print(
            f"  {target:20s} legacy={legacy_total:4d} rework={rework_total:4d} diff={diff:4d} ({status})"
        )


def detail_target(target: str, label: str, sequences: Iterable[SequenceInfo]) -> Counter:
    subset = [seq for seq in sequences if seq.target == target]
    print(f"\n{label} sequences for {target}: {len(subset)}")
    if not subset:
        return Counter()

    for seq in sorted(subset, key=lambda item: (item.start, item.visit_id, item.seq_id)):
        print(
            f"  {seq.visit_id}-{seq.seq_id:>03s} {seq.start} -> {seq.stop} priority={seq.priority}"
        )

    return Counter((seq.start, seq.stop, seq.priority) for seq in subset)


def show_target_differences(target: str, legacy_data: CalendarData, rework_data: CalendarData) -> None:
    legacy_counter = detail_target(target, "Legacy", legacy_data.sequences)
    rework_counter = detail_target(target, "Rework", rework_data.sequences)

    missing = legacy_counter - rework_counter
    extra = rework_counter - legacy_counter

    if missing:
        print("\nSequences missing from rework:")
        for (start, stop, priority), count in missing.items():
            print(f"  ({count}x) {start} -> {stop} priority={priority}")

    if extra:
        print("\nSequences extra in rework:")
        for (start, stop, priority), count in extra.items():
            print(f"  ({count}x) {start} -> {stop} priority={priority}")

    if not missing and not extra:
        print("\nAll sequences match for this target.")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_arguments(argv)

    legacy_data = load_calendar(args.legacy)
    rework_data = load_calendar(args.rework)

    legacy_counts = summarise_counts("Legacy", legacy_data)
    rework_counts = summarise_counts("Rework", rework_data)

    differences = compute_differences(legacy_counts, rework_counts)
    show_mismatches(differences, args.top, args.include_equal)

    if args.target:
        show_target_differences(args.target, legacy_data, rework_data)


if __name__ == "__main__":
    main()
