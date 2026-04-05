# 01_SDD-YOLO

## Module Identity
- Module: `DEF-sdd-yolo`
- Scope: Ground-to-air micro-UAV detection from RGB imagery.
- Primary source: `papers/2603.25218.pdf` (SDD-YOLO).

## Mission
Implement an ANIMA-ready, edge-efficient SDD-YOLO baseline that is faithful to the paper's core claims:
- P2 (4x) high-resolution detection head for micro-targets.
- DFL-free regression path.
- NMS-free one-to-one inference mode.
- Dual attention for aerial clutter suppression.
- Training recipe hooks for MuSGD + STAL style supervision.

## Current Status
- Paper parsed locally.
- No official upstream repo or dataset release link found in paper/arXiv source.
- This repository contains implementation scaffold and reproducible local baseline code.

## Local Conventions
- Python package root: `src/sdd_yolo/`
- Configs: `configs/*.toml`
- CLIs: `scripts/train.py`, `scripts/infer.py`, `scripts/eval.py`, `scripts/export_onnx.py`
- Tests: `tests/`
