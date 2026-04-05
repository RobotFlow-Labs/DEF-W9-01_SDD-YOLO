# DEF-sdd-yolo

Paper-driven ANIMA defense module for G2A micro-UAV detection based on:
- `papers/2603.25218.pdf`

## Quickstart
```bash
uv sync --dev
uv run pytest -q
uv run python scripts/infer.py --help
```

## Structure
- `ASSETS.md` — paper assets + expected metrics
- `prds/` — 7-PRD build plan
- `tasks/` — granular implementation tasks
- `src/sdd_yolo/` — implementation
