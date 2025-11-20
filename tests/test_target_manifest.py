from __future__ import annotations

from importlib import util as importlib_util
from pathlib import Path
import types

import warnings

import pandas as pd
import pytest
from erfa import ErfaWarning


warnings.filterwarnings("ignore", category=ErfaWarning)

pytestmark = pytest.mark.filterwarnings(
    "ignore:ERFA function\ .*:erfa.ErfaWarning"
)

from pandorascheduler_rework.targets.manifest import build_target_manifest


BASE_DIR = (
    Path(__file__).resolve().parents[1]
    / "comparison_outputs"
    / "target_definition_files_limited"
)


def _load_legacy_helper_codes() -> types.ModuleType:
    module_name = "_legacy_helper_codes_for_manifest_test"
    legacy_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "pandorascheduler"
        / "helper_codes.py"
    )
    spec = importlib_util.spec_from_file_location(module_name, legacy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load legacy helper_codes module for tests")
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy_helper = _load_legacy_helper_codes()


@pytest.mark.parametrize(
    "category",
    [
        "exoplanet",
        "primary-exoplanet",
        "secondary-exoplanet",
        "auxiliary-exoplanet",
        "auxiliary-standard",
        "monitoring-standard",
        "occultation-standard",
    ],
)
def test_manifest_matches_legacy(category: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if not BASE_DIR.is_dir():
        pytest.skip("Target definition fixtures are not available")

    monkeypatch.setattr(legacy_helper, "TARGET_DEF_BASE", str(BASE_DIR))
    monkeypatch.setattr(legacy_helper, "tqdm", lambda iterable, **_: iterable)

    legacy_df = legacy_helper.process_target_files(category)
    rework_df = build_target_manifest(category, BASE_DIR)

    assert isinstance(legacy_df, pd.DataFrame)
    assert isinstance(rework_df, pd.DataFrame)

    assert set(legacy_df.columns) == set(rework_df.columns)

    rework_df = rework_df[legacy_df.columns]

    sort_key = "Original Filename" if "Original Filename" in legacy_df.columns else "Star Name"
    legacy_sorted = legacy_df.sort_values(by=sort_key).reset_index(drop=True)
    rework_sorted = rework_df.sort_values(by=sort_key).reset_index(drop=True)

    pd.testing.assert_frame_equal(rework_sorted, legacy_sorted)
