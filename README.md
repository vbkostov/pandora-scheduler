# Pandora Scheduler (rework)

Brief overview and quick links for developers working on the rework.

- See `QUICK_START.md` for runnable examples and common workflows.
- Example JSON configuration: `example_scheduler_config.json` (root) — use with `--config`.
- Detailed keys: `docs/EXAMPLE_SCHEDULER_CONFIG.md`.

Quick start (two commands):

```fish
# 1) install deps (poetry environment assumed)
poetry install

# 2) run a quick test (assumes manifests/visibility already present)
poetry run python run_scheduler.py --start "2026-02-05" --end "2026-02-07" --output ./output_test
```

If you need help, read `QUICK_START.md` for examples and troubleshooting tips.
