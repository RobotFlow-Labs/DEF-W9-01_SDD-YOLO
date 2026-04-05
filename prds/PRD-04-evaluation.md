# PRD-04: Evaluation and Benchmark Comparison

> Module: DEF-sdd-yolo | Priority: P1
> Depends on: PRD-03
> Status: ⬜ Not started

## Objective
Implement evaluation scripts for mAP metrics and reproducible benchmark reporting against paper targets.

## Context (from paper)
Primary metrics: mAP@0.5, mAP@0.5:0.95, CPU FPS, GPU FPS.
Paper reference: Table 2, Table 3, Table 4, Section 5.

## Acceptance Criteria
- [ ] Evaluate predictions against YOLO labels (COCO-style AP approximation acceptable for MVP).
- [ ] Produce JSON/Markdown summary report.
- [ ] Compare results to paper target table.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---:|
| `src/sdd_yolo/eval.py` | metric engine and reporting | §5 | ~260 |
| `scripts/eval.py` | evaluation CLI | — | ~60 |
| `benchmarks/paper_targets.json` | expected benchmark values | Table 2-4 | ~50 |
| `tests/test_metrics.py` | IoU/AP metric unit tests | — | ~120 |

## Test Plan
```bash
uv run pytest tests/test_metrics.py -v
uv run python scripts/eval.py --help
```

## References
- Paper: Section 5, Tables 2/3/4
- Feeds into: PRD-07
