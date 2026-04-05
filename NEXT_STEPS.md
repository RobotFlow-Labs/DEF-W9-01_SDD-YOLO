# NEXT_STEPS — DEF-sdd-yolo

## Current Checkpoint (2026-04-04)
- [x] Paper parsed (`2603.25218`).
- [x] Asset manifest drafted from paper claims and available infra context.
- [x] PRD suite and granular tasks generated.
- [x] Implement essential local code scaffold (model, train/eval/infer, API, ROS2 stubs, export).
- [x] Run full local test suite after implementation changes.
- [x] Validate CLI smoke commands (`infer.py`, `eval.py`, `serve.py` help/boot path).
- [ ] Validate on real DroneSOD-30K paths when available.
- [ ] Execute CUDA-server optimization and training pipeline.

## Active Blockers
- DroneSOD-30K dataset path and official release package are not available in this repo.
- YOLO26 official checkpoint naming/path needs confirmation during server run.

## Resume Procedure
1. `source .venv311/bin/activate` (or `uv sync --dev` if you prefer uv-managed env).
2. `python -m pytest -q`
3. `python scripts/infer.py --help`
4. Provision DroneSOD-30K paths and run training/evaluation on CUDA server.
