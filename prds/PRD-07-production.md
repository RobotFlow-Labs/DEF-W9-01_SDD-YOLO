# PRD-07: Production Hardening and Export Pipeline

> Module: DEF-sdd-yolo | Priority: P2
> Depends on: PRD-04, PRD-05, PRD-06
> Status: ⬜ Not started

## Objective
Prepare production export path (ONNX/TRT), reliability checks, and ops documentation for server rollout.

## Context (from paper)
Paper highlights edge deployment and quantization sensitivity (DFL removal rationale).
Paper reference: Section 4.3, 5.4, 6 future work.

## Acceptance Criteria
- [ ] ONNX export script works for model checkpoint.
- [ ] Export report includes model I/O schema and dynamic axes.
- [ ] Runtime checklist captures INT8 risk points and deployment constraints.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---:|
| `src/sdd_yolo/export.py` | ONNX export helper | §4.3/§6 | ~160 |
| `scripts/export_onnx.py` | export CLI | — | ~50 |
| `docs/deployment_checklist.md` | prod readiness checklist | §5.4/§6 | ~120 |
| `docs/training_report_template.md` | benchmark report template | §5 | ~80 |

## Test Plan
```bash
uv run python scripts/export_onnx.py --help
```

## References
- Paper: Sections 4.3, 5.4, 6
