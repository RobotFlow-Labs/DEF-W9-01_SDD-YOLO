# DEF-sdd-yolo — Build Plan

## Objective
Deliver a paper-faithful, local-runnable SDD-YOLO baseline with PRD-driven implementation, then prepare training/export hooks for CUDA-server hardening.

## PRD Execution Board
| PRD | Title | Priority | Status | Notes |
|---|---|---|---|---|
| PRD-01 | Foundation & Config | P0 | COMPLETE_LOCAL | Core scaffolding and config contracts |
| PRD-02 | Core Model | P0 | COMPLETE_LOCAL | P2 head + dual attention + DFL-free head |
| PRD-03 | Inference Pipeline | P0 | COMPLETE_LOCAL | Image inference + optional NMS-free path |
| PRD-04 | Evaluation | P1 | COMPLETE_LOCAL | mAP/FPS evaluation utilities |
| PRD-05 | API & Docker | P1 | COMPLETE_LOCAL | FastAPI endpoint + serving container files |
| PRD-06 | ROS2 Integration | P1 | COMPLETE_LOCAL | Interface stubs for ANIMA ROS graph |
| PRD-07 | Production Hardening | P2 | COMPLETE_LOCAL | ONNX export helpers + deployment/checklist docs |

## Constraints
- No official upstream repo available: implementation is built from paper specification.
- DroneSOD-30K binaries are not bundled in this repo.
- Training on real data is blocked until dataset paths are provisioned.

## Definition of Done (MVP)
- [x] Package installs and imports cleanly.
- [x] Model forward pass and one training step execute.
- [x] Inference CLI entrypoint is functional.
- [x] Evaluation CLI entrypoint is functional.
- [x] API `/health` and `/predict` contract is implemented and tested.
