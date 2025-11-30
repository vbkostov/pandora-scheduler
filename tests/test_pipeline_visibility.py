from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pandorascheduler_rework.pipeline import (
    SchedulerPaths,
    SchedulerRequest,
    _build_visibility_config,
    _maybe_generate_visibility,
)
from pandorascheduler_rework.visibility.config import VisibilityConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_maybe_generate_visibility_invokes_builder(monkeypatch, tmp_path):
    repo_root = _repo_root()
    package_root = (repo_root / "src" / "pandorascheduler").resolve()
    paths = SchedulerPaths.from_package_root(package_root)

    primary_target_csv = (paths.data_dir / "exoplanet_targets.csv").resolve()
    auxiliary_target_csv = (paths.data_dir / "auxiliary-standard_targets.csv").resolve()
    occultation_target_csv = (paths.data_dir / "occultation-standard_targets.csv").resolve()

    request = SchedulerRequest(
        targets_manifest=package_root / "data" / "baseline" / "fingerprints.json",
        window_start=datetime(2026, 2, 5),
        window_end=datetime(2027, 2, 5),
        output_dir=tmp_path,
        config={"generate_visibility": True},
    )

    captured_configs: list[VisibilityConfig] = []

    def fake_builder(cfg):  # type: ignore[no-untyped-def]
        captured_configs.append(cfg)

    monkeypatch.setattr(
        "pandorascheduler_rework.pipeline.build_visibility_catalog",
        fake_builder,
    )

    _maybe_generate_visibility(
        request,
        paths,
        request.window_start,
        request.window_end,
        primary_target_csv,
        auxiliary_target_csv,
        occultation_target_csv,
    )

    assert len(captured_configs) == 3
    # The first call should be for primary targets
    visibility_cfg = captured_configs[0]
    assert isinstance(visibility_cfg, VisibilityConfig)
    assert visibility_cfg.window_start == request.window_start
    assert visibility_cfg.window_end == request.window_end
    assert visibility_cfg.target_list == primary_target_csv
    # Default GMAT file should live under the legacy data directory; use the
    # repo layout rather than relying on the current working directory.
    expected_gmat = (paths.data_dir / "Pandora-600km-withoutdrag-20251018.txt").resolve()
    assert visibility_cfg.gmat_ephemeris == expected_gmat
    assert visibility_cfg.partner_list == auxiliary_target_csv
    assert not visibility_cfg.force
    assert visibility_cfg.target_filters == ()


def test_maybe_generate_visibility_skips_without_flag(monkeypatch, tmp_path):
    repo_root = _repo_root()
    package_root = (repo_root / "src" / "pandorascheduler").resolve()
    paths = SchedulerPaths.from_package_root(package_root)

    primary_target_csv = (paths.data_dir / "exoplanet_targets.csv").resolve()
    auxiliary_target_csv = (paths.data_dir / "auxiliary-standard_targets.csv").resolve()
    occultation_target_csv = (paths.data_dir / "occultation-standard_targets.csv").resolve()

    request = SchedulerRequest(
        targets_manifest=package_root / "data" / "baseline" / "fingerprints.json",
        window_start=datetime(2026, 2, 5),
        window_end=datetime(2027, 2, 5),
        output_dir=tmp_path,
    )

    called = False

    def fake_builder(_cfg):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True

    monkeypatch.setattr(
        "pandorascheduler_rework.pipeline.build_visibility_catalog",
        fake_builder,
    )

    _maybe_generate_visibility(
        request,
        paths,
        request.window_start,
        request.window_end,
        primary_target_csv,
        auxiliary_target_csv,
        occultation_target_csv,
    )

    assert not called


def test_build_visibility_config_supports_overrides(tmp_path):
    repo_root = _repo_root()
    package_root = (repo_root / "src" / "pandorascheduler").resolve()
    paths = SchedulerPaths.from_package_root(package_root)

    primary_target_csv = (paths.data_dir / "exoplanet_targets.csv").resolve()
    auxiliary_target_csv = (paths.data_dir / "auxiliary-standard_targets.csv").resolve()

    custom_gmat = tmp_path / "custom_gmat.csv"
    custom_gmat.write_text("time,values\n")

    custom_targets = tmp_path / "custom_targets.csv"
    custom_targets.write_text("Star Name,Star Simbad Name\n")

    custom_output = tmp_path / "vis_outputs"

    request = SchedulerRequest(
        targets_manifest=package_root / "data" / "baseline" / "fingerprints.json",
        window_start=datetime(2026, 1, 1),
        window_end=datetime(2026, 12, 31),
        output_dir=tmp_path,
        config={
            "generate_visibility": True,
            "visibility_gmat": str(custom_gmat),
            "visibility_target_list": str(custom_targets),
            "visibility_partner_list": "",
            "visibility_output_root": str(custom_output),
            "visibility_sun_deg": "95",
            "visibility_moon_deg": 35,
            "visibility_earth_deg": "75",
            "visibility_force": True,
            "visibility_target_filters": "Alpha, Beta",
        },
    )

    visibility_config = _build_visibility_config(
        request,
        paths,
        request.window_start,
        request.window_end,
        target_list=primary_target_csv,
        partner_list=auxiliary_target_csv,
        output_subpath="targets",
    )

    assert visibility_config.gmat_ephemeris == custom_gmat.resolve()
    assert visibility_config.target_list == custom_targets.resolve()
    assert visibility_config.partner_list is None
    assert visibility_config.output_root == custom_output.resolve()
    assert visibility_config.sun_avoidance_deg == 95.0
    assert visibility_config.moon_avoidance_deg == 35.0
    assert visibility_config.earth_avoidance_deg == 75.0
    assert visibility_config.force is True
    assert visibility_config.target_filters == ("Alpha", "Beta")