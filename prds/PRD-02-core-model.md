# PRD-02: Core SDD-YOLO Model (P2 + Dual Attention + DFL-free Head)

> Module: DEF-sdd-yolo | Priority: P0
> Depends on: PRD-01
> Status: ⬜ Not started

## Objective
Implement the essential SDD-YOLO architecture components described in the paper.

## Context (from paper)
Key novelty is a P2 head at 4x downsampling to preserve sub-16-pixel targets, plus dual attention and DFL-free regression.
Paper reference: Section 4.2, 4.3, 4.5, Eq. (1)-(4).

## Acceptance Criteria
- [ ] Model produces multi-scale outputs with explicit P2 branch.
- [ ] Dual attention module can be toggled and unit-tested.
- [ ] DFL-free detection head path is default.
- [ ] One training step runs with finite loss.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---:|
| `src/sdd_yolo/models/blocks.py` | CSP/conv/FPN blocks | §4.1-4.2 | ~220 |
| `src/sdd_yolo/models/attention.py` | dual attention implementation | Eq. (4) | ~120 |
| `src/sdd_yolo/models/head.py` | DFL-free head and decode | §4.3-4.4 | ~180 |
| `src/sdd_yolo/models/model.py` | full SDDYOLO model assembly | §4.1-4.5 | ~260 |
| `src/sdd_yolo/losses.py` | IoU + cls/objectness losses, STAL weights | §4.3/4.6 | ~220 |
| `tests/test_model_forward.py` | shape + finite-loss tests | — | ~130 |

## Test Plan
```bash
uv run pytest tests/test_model_forward.py -v
```

## References
- Paper: Section 4.1-4.6
- Feeds into: PRD-03
