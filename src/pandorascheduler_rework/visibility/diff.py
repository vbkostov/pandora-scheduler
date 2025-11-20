"""Helpers for comparing visibility output trees."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

import numpy as np
import pandas as pd

DEFAULT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class FileComparison:
    """Detailed results for a single CSV comparison."""

    path: Path
    shape_mismatch: tuple[tuple[int, int], tuple[int, int]] | None = None
    extra_columns_legacy: Sequence[str] = field(default_factory=tuple)
    extra_columns_rework: Sequence[str] = field(default_factory=tuple)
    visible_mismatches: int = 0
    numeric_deltas: Dict[str, float] = field(default_factory=dict)
    non_numeric_mismatches: Dict[str, int] = field(default_factory=dict)

    @property
    def is_match(self) -> bool:
        return (
            self.shape_mismatch is None
            and not self.extra_columns_legacy
            and not self.extra_columns_rework
            and self.visible_mismatches == 0
            and not self.numeric_deltas
            and not self.non_numeric_mismatches
        )


@dataclass(frozen=True)
class ComparisonSummary:
    """Aggregated comparison results for a full directory tree."""

    missing_in_legacy: Sequence[Path]
    missing_in_rework: Sequence[Path]
    file_results: Sequence[FileComparison]

    @property
    def differing_files(self) -> List[FileComparison]:
        return [result for result in self.file_results if not result.is_match]

    @property
    def identical(self) -> bool:
        return not self.missing_in_legacy and not self.missing_in_rework and not self.differing_files


def _iter_csv_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*.csv")):
        if path.is_file():
            yield path


def compare_visibility_trees(
    legacy_root: Path,
    rework_root: Path,
    *,
    atol: float = DEFAULT_TOLERANCE,
) -> ComparisonSummary:
    """Compare two visibility directory trees.

    Parameters
    ----------
    legacy_root
        Root directory containing the legacy CSV outputs.
    rework_root
        Root directory containing the rework CSV outputs.
    atol
        Absolute tolerance to use when comparing floating-point columns.
    """

    legacy_paths = {path.relative_to(legacy_root) for path in _iter_csv_files(legacy_root)}
    rework_paths = {path.relative_to(rework_root) for path in _iter_csv_files(rework_root)}

    missing_in_legacy = sorted(rework_paths - legacy_paths)
    missing_in_rework = sorted(legacy_paths - rework_paths)

    file_results: List[FileComparison] = []

    for rel_path in sorted(legacy_paths & rework_paths):
        legacy_path = legacy_root / rel_path
        rework_path = rework_root / rel_path
        comparison = _compare_csv_files(legacy_path, rework_path, rel_path, atol=atol)
        file_results.append(comparison)

    return ComparisonSummary(
        missing_in_legacy=missing_in_legacy,
        missing_in_rework=missing_in_rework,
        file_results=file_results,
    )


def _compare_csv_files(
    legacy_path: Path,
    rework_path: Path,
    rel_path: Path,
    *,
    atol: float,
) -> FileComparison:
    legacy_df = pd.read_csv(legacy_path)
    rework_df = pd.read_csv(rework_path)

    shape_mismatch = None
    if legacy_df.shape != rework_df.shape:
        shape_mismatch = (legacy_df.shape, rework_df.shape)

    if legacy_df.shape[0] != rework_df.shape[0]:
        return FileComparison(
            path=rel_path,
            shape_mismatch=shape_mismatch,
            extra_columns_legacy=_extra_columns(legacy_df, rework_df),
            extra_columns_rework=_extra_columns(rework_df, legacy_df),
        )

    extra_legacy = _extra_columns(legacy_df, rework_df)
    extra_rework = _extra_columns(rework_df, legacy_df)

    visible_mismatches = 0
    numeric_deltas: Dict[str, float] = {}
    non_numeric_mismatches: Dict[str, int] = {}

    common_columns = [column for column in legacy_df.columns if column in rework_df.columns]

    for column in common_columns:
        legacy_series = legacy_df[column]
        rework_series = rework_df[column]

        if column == "Visible":
            mismatches = int((legacy_series != rework_series).sum())
            if mismatches:
                visible_mismatches = mismatches
            continue

        if pd.api.types.is_numeric_dtype(legacy_series) and pd.api.types.is_numeric_dtype(rework_series):
            delta = np.nanmax(np.abs(legacy_series.to_numpy(dtype=float) - rework_series.to_numpy(dtype=float)))
            if not np.isnan(delta) and delta > atol:
                numeric_deltas[column] = float(delta)
            continue

        mismatches = int((legacy_series.fillna("") != rework_series.fillna("")).sum())
        if mismatches:
            non_numeric_mismatches[column] = mismatches

    return FileComparison(
        path=rel_path,
        shape_mismatch=shape_mismatch,
        extra_columns_legacy=extra_legacy,
        extra_columns_rework=extra_rework,
        visible_mismatches=visible_mismatches,
        numeric_deltas=numeric_deltas,
        non_numeric_mismatches=non_numeric_mismatches,
    )


def _extra_columns(primary: pd.DataFrame, other: pd.DataFrame) -> Sequence[str]:
    extras = [column for column in primary.columns if column not in other.columns]
    return tuple(extras)


def _format_summary(summary: ComparisonSummary) -> str:
    lines: List[str] = []
    if summary.missing_in_legacy:
        lines.append("Files missing in legacy:")
        lines.extend(f"  {path}" for path in summary.missing_in_legacy)
    if summary.missing_in_rework:
        lines.append("Files missing in rework:")
        lines.extend(f"  {path}" for path in summary.missing_in_rework)
    differing = summary.differing_files
    lines.append(f"Total differing files: {len(differing)}")
    for result in differing:
        lines.append(f"--- {result.path}")
        if result.shape_mismatch is not None:
            lines.append(
                "  shape mismatch: "
                f"legacy={result.shape_mismatch[0]} rework={result.shape_mismatch[1]}"
            )
        if result.extra_columns_legacy:
            lines.append(f"  columns only in legacy: {sorted(result.extra_columns_legacy)}")
        if result.extra_columns_rework:
            lines.append(f"  columns only in rework: {sorted(result.extra_columns_rework)}")
        if result.visible_mismatches:
            lines.append(f"  visible mismatches: {result.visible_mismatches}")
        for column, delta in result.numeric_deltas.items():
            lines.append(f"  max |Δ {column}| = {delta:.6g}")
        for column, count in result.non_numeric_mismatches.items():
            lines.append(f"  non-numeric mismatches in {column}: {count}")
    return "\n".join(lines)


def compare_and_print(
    legacy_root: Path,
    rework_root: Path,
    *,
    atol: float = DEFAULT_TOLERANCE,
) -> ComparisonSummary:
    summary = compare_visibility_trees(legacy_root, rework_root, atol=atol)
    print(_format_summary(summary))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Compare visibility CSV outputs.")
    parser.add_argument("legacy", type=Path, help="Legacy visibility root directory")
    parser.add_argument("rework", type=Path, help="Rework visibility root directory")
    parser.add_argument("--atol", type=float, default=DEFAULT_TOLERANCE, help="Absolute tolerance for numeric comparisons")
    args = parser.parse_args(argv)

    summary = compare_and_print(args.legacy, args.rework, atol=args.atol)
    return 0 if summary.identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
