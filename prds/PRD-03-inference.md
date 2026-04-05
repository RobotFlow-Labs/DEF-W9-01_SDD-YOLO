# PRD-03: Inference Pipeline and CLI

> Module: DEF-sdd-yolo | Priority: P0
> Depends on: PRD-02
> Status: ⬜ Not started

## Objective
Provide local inference tools for images with optional NMS-free decode mode and profiling hooks.

## Context (from paper)
Paper emphasizes end-to-end efficiency with NMS-free inference and real-time FPS reporting.
Paper reference: Section 4.4 and 5.2/5.4.

## Acceptance Criteria
- [ ] CLI inference works on image folder/input glob.
- [ ] Emits detections and annotated previews.
- [ ] Supports `--nms-free` and threshold tuning.
- [ ] Reports average latency/FPS.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---:|
| `src/sdd_yolo/infer.py` | core inference runner | §4.4, §5.4 | ~220 |
| `scripts/infer.py` | CLI entrypoint | — | ~50 |
| `src/sdd_yolo/visualize.py` | render boxes for inspection | §5.4 qual examples | ~120 |
| `tests/test_inference_smoke.py` | smoke inference test | — | ~90 |

## Test Plan
```bash
uv run pytest tests/test_inference_smoke.py -v
uv run python scripts/infer.py --help
```

## References
- Paper: Section 4.4, 5.2, 5.4
- Feeds into: PRD-04 and PRD-05
