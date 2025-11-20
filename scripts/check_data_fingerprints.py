"""Utilities for recording and comparing data product fingerprints.

This script helps validate that regenerated scheduling inputs match a prior
baseline without checking large CSV visibility tables into source control.

Usage examples (run from repository root):

    # Record hashes for the default data products into baseline/fingerprints.json
    python scripts/check_data_fingerprints.py snapshot

    # Compare current files against the stored manifest
    python scripts/check_data_fingerprints.py compare

    # Print hashes without touching disk
    python scripts/check_data_fingerprints.py show --output -

You can override the manifest location or the list of files via CLI flags.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "src" / "pandorascheduler" / "data"
DEFAULT_MANIFEST = DATA_ROOT / "baseline" / "fingerprints.json"

# Small text files we always want to track
DEFAULT_TARGET_CSVS = [
    DATA_ROOT / "exoplanet_targets.csv",
    DATA_ROOT / "auxiliary-standard_targets.csv",
    DATA_ROOT / "monitoring-standard_targets.csv",
    DATA_ROOT / "occultation-standard_targets.csv",
]

# Larger visibility directories – we hash individual files so the manifest
# remains relatively small while still catching differences.
DEFAULT_DIRECTORIES = [
    DATA_ROOT / "targets",
    DATA_ROOT / "aux_targets",
]


def compute_sha256(path: Path) -> str:
    """Return the hex digest for *path* using SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(paths: Iterable[Path]) -> Iterable[Tuple[Path, Path]]:
    """Yield (absolute_path, relative_path) pairs for the requested *paths*."""
    for entry in paths:
        entry = entry.resolve()
        if entry.is_file():
            yield entry, entry.relative_to(DATA_ROOT)
            continue
        if entry.is_dir():
            for file_path in sorted(entry.rglob("*")):
                if file_path.is_file():
                    yield file_path, file_path.relative_to(DATA_ROOT)


def build_manifest(files: Iterable[Path], dirs: Iterable[Path]) -> Dict[str, str]:
    """Hash the provided files/directories and return a mapping."""
    manifest: Dict[str, str] = {}
    for abs_path, rel_path in iter_files(list(files) + list(dirs)):
        manifest[str(rel_path)] = compute_sha256(abs_path)
    return dict(sorted(manifest.items()))


def write_manifest(manifest: Dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"root": str(DATA_ROOT), "hashes": manifest}, handle, indent=2)
        handle.write("\n")


def load_manifest(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("hashes", {})


def compare_manifests(a: Dict[str, str], b: Dict[str, str]) -> Tuple[List[str], List[str], List[str]]:
    """Compare two manifests; return (added, removed, changed) lists."""
    added = sorted(set(b) - set(a))
    removed = sorted(set(a) - set(b))
    changed = sorted(key for key in set(a) & set(b) if a[key] != b[key])
    return added, removed, changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--manifest",
            type=Path,
            default=DEFAULT_MANIFEST,
            help=f"Manifest file location (default: {DEFAULT_MANIFEST.relative_to(REPO_ROOT)})",
        )
        subparser.add_argument(
            "--files",
            nargs="*",
            type=Path,
            default=[*DEFAULT_TARGET_CSVS],
            help="Specific files to hash (defaults to target CSVs).",
        )
        subparser.add_argument(
            "--dirs",
            nargs="*",
            type=Path,
            default=[*DEFAULT_DIRECTORIES],
            help="Directories to hash recursively (defaults to visibility trees).",
        )
        subparser.add_argument(
            "--output",
            type=Path,
            default=None,
            help="Write manifest to this path instead of the manifest file (use '-' for stdout).",
        )

    add_common(subparsers.add_parser("snapshot", help="Compute and store a fresh manifest."))
    add_common(subparsers.add_parser("compare", help="Compare current files against an existing manifest."))
    add_common(subparsers.add_parser("show", help="Compute hashes and print them without storing."))

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    files = [path if path.is_absolute() else DATA_ROOT / path for path in args.files]
    dirs = [path if path.is_absolute() else DATA_ROOT / path for path in args.dirs]

    manifest = build_manifest(files, dirs)

    if args.command == "show":
        target = args.output
        if target is None or str(target) == "-":
            for rel_path, digest in manifest.items():
                print(f"{digest}  {rel_path}")
        else:
            write_manifest(manifest, target)
        return

    if args.command == "snapshot":
        output_path = args.output or args.manifest
        if output_path == Path("-"):
            for rel_path, digest in manifest.items():
                print(f"{digest}  {rel_path}")
        else:
            write_manifest(manifest, output_path)
            print(f"Wrote manifest with {len(manifest)} entries to {output_path}")
        return

    if args.command == "compare":
        reference_path = args.output or args.manifest
        if reference_path == Path("-"):
            raise SystemExit("--output - is not supported for compare")
        if not reference_path.exists():
            raise SystemExit(f"Reference manifest not found: {reference_path}")
        reference_manifest = load_manifest(reference_path)
        added, removed, changed = compare_manifests(reference_manifest, manifest)

        if not (added or removed or changed):
            print(f"No differences detected across {len(manifest)} entries.")
            return

        if added:
            print("Added files:")
            for key in added:
                print(f"  + {key}")
        if removed:
            print("Removed files:")
            for key in removed:
                print(f"  - {key}")
        if changed:
            print("Changed files:")
            for key in changed:
                print(f"  * {key}")
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
