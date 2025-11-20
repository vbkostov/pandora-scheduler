"""Helpers to compare reworked XML output against the legacy script."""

from __future__ import annotations

import difflib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from pandorascheduler_rework import observation_utils, science_calendar


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of a rework vs legacy XML comparison."""

    match: bool
    legacy_xml: str
    rework_xml: str
    diff: str


def compare_with_legacy(
    *,
    legacy_package_dir: Optional[Path] = None,
    schedule_filename: str = "Pandora_Schedule_0.8_0.0_0.2_2026-02-05_to_2027-02-05.csv",
    python_executable: str = sys.executable,
    extra_pythonpath: Optional[Iterable[str]] = None,
    schedule_csv_path: Optional[Path] = None,
) -> ComparisonResult:
    """Run the legacy generator and the rework builder and compare their outputs.

    Parameters
    ----------
    legacy_package_dir
        Path to the legacy :mod:`pandorascheduler` package root.  Defaults to the
        repository copy under ``src/pandorascheduler``.
    schedule_filename
        CSV schedule filename relative to ``legacy_package_dir / "data"`` that
        the generators should consume.
    python_executable
        Python interpreter used to execute the legacy script.  Defaults to the
        interpreter running the rework code.
    extra_pythonpath
        Iterable of paths appended to the ``PYTHONPATH`` when executing the
        legacy script.  This can help when running from within alternative
        environments.
    """

    repo_root = Path(__file__).resolve().parents[2]
    legacy_package_dir = legacy_package_dir or (repo_root / "src" / "pandorascheduler")
    if not legacy_package_dir.is_dir():
        raise FileNotFoundError(f"Legacy package directory not found: {legacy_package_dir}")

    data_dir = legacy_package_dir / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Legacy data directory not found: {data_dir}")

    schedule_csv_source = Path(schedule_csv_path) if schedule_csv_path else None
    if schedule_csv_source is not None and not schedule_csv_source.is_file():
        raise FileNotFoundError(f"Provided schedule CSV not found: {schedule_csv_source}")

    default_schedule_csv = data_dir / schedule_filename
    if schedule_csv_source is None and not default_schedule_csv.is_file():
        raise FileNotFoundError(f"Schedule CSV not found: {default_schedule_csv}")

    with tempfile.TemporaryDirectory() as tmp_root_str:
        tmp_root = Path(tmp_root_str)
        legacy_copy = tmp_root / "legacy"
        shutil.copytree(legacy_package_dir, legacy_copy)

        legacy_env = os.environ.copy()
        pythonpath_value = legacy_env.get("PYTHONPATH", "")
        pythonpath_parts = [part for part in pythonpath_value.split(os.pathsep) if part]
        if extra_pythonpath:
            pythonpath_parts.extend(str(Path(p)) for p in extra_pythonpath)
        pythonpath_parts.append(str(legacy_copy))
        legacy_env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

        legacy_schedule_csv = legacy_copy / "data" / schedule_filename
        if schedule_csv_source is not None:
            shutil.copy2(schedule_csv_source, legacy_schedule_csv)
        elif not legacy_schedule_csv.is_file():
            raise FileNotFoundError(f"Schedule CSV missing in legacy copy: {legacy_schedule_csv}")

        default_schedule_name = "Pandora_Schedule_0.8_0.0_0.2_2026-02-05_to_2027-02-05.csv"
        default_schedule_csv = legacy_copy / "data" / default_schedule_name
        if not default_schedule_csv.exists() or default_schedule_csv != legacy_schedule_csv:
            shutil.copy2(legacy_schedule_csv, default_schedule_csv)

        try:
            subprocess.run(
                [python_executable, "sched2xml_WIP.py"],
                cwd=legacy_copy,
                check=True,
                capture_output=True,
                text=True,
                env=legacy_env,
            )
        except subprocess.CalledProcessError as exc:
            message = [
                "Legacy XML generation failed",
                f"stdout:\n{exc.stdout.strip()}" if exc.stdout else "stdout: <empty>",
                f"stderr:\n{exc.stderr.strip()}" if exc.stderr else "stderr: <empty>",
            ]
            raise RuntimeError("\n".join(message)) from exc
        legacy_output_path = legacy_copy / "data" / "Pandora_science_calendar.xml"
        legacy_xml_text = legacy_output_path.read_text(encoding="utf-8")

        created_override = _extract_created_timestamp(legacy_xml_text)

        rework_output_path = tmp_root / "rework_science_calendar.xml"

        original_data_roots = observation_utils.DATA_ROOTS
        observation_utils.DATA_ROOTS = [legacy_copy / "data"]
        try:
            inputs = science_calendar.ScienceCalendarInputs(
                schedule_csv=legacy_schedule_csv,
                data_dir=legacy_copy / "data",
            )
            calendar_config = science_calendar.ScienceCalendarConfig(
                created_timestamp=created_override,
                show_progress=True,
            )
            science_calendar.generate_science_calendar(
                inputs,
                output_path=rework_output_path,
                config=calendar_config,
            )
        finally:
            observation_utils.DATA_ROOTS = original_data_roots

        rework_xml_text = rework_output_path.read_text(encoding="utf-8")

        if legacy_xml_text == rework_xml_text:
            return ComparisonResult(True, legacy_xml_text, rework_xml_text, diff="")

        diff_lines = difflib.unified_diff(
            legacy_xml_text.splitlines(),
            rework_xml_text.splitlines(),
            fromfile="legacy",
            tofile="rework",
            lineterm="",
        )
        diff_text = "\n".join(diff_lines)
        return ComparisonResult(False, legacy_xml_text, rework_xml_text, diff=diff_text)


def _extract_created_timestamp(xml_text: str) -> str | None:
    match = re.search(r'Created="([^"]+)"', xml_text)
    if match:
        return match.group(1)
    return None