# PRD-05: API Serving and Docker Runtime

> Module: DEF-sdd-yolo | Priority: P1
> Depends on: PRD-03
> Status: ⬜ Not started

## Objective
Expose a deployable inference service with health/readiness and prediction endpoints.

## Context (from paper)
Deployment focus is edge efficiency and streamlined inference graph.
Paper reference: Abstract, Section 1, Section 5.4.

## Acceptance Criteria
- [ ] FastAPI app starts and returns `/health`.
- [ ] `/predict` accepts image upload and returns detection JSON.
- [ ] Dockerfile and compose file build/start cleanly for local serving.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---:|
| `src/sdd_yolo/serve.py` | FastAPI app | deployment context | ~180 |
| `scripts/serve.py` | local API runner | — | ~40 |
| `Dockerfile.serve` | container image | — | ~80 |
| `docker-compose.serve.yml` | local orchestration | — | ~60 |
| `tests/test_api.py` | API unit/smoke test | — | ~100 |

## Test Plan
```bash
uv run pytest tests/test_api.py -v
uv run python scripts/serve.py --help
```

## References
- Paper: deployment motivation throughout intro/experiments
- Feeds into: PRD-06, PRD-07
