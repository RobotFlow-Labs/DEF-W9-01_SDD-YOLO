# PRD-01: Foundation, Config, and Data Contracts

> Module: DEF-sdd-yolo | Priority: P0
> Depends on: None
> Status: ⬜ Not started

## Objective
Create a reproducible foundation (package, config, dataset I/O, sanity tests) for SDD-YOLO development.

## Context (from paper)
The paper relies on YOLO-format annotations and a dedicated dataset split (DroneSOD-30K) with severe micro-target imbalance.
Paper reference: Section 3.1-3.2.

## Acceptance Criteria
- [ ] Package installs with `uv sync`.
- [ ] Configs load (`default`, `paper`, `debug`).
- [ ] YOLO-format dataset loader returns image + targets tensors.
- [ ] Unit tests pass for config and dataset parsing.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---:|
| `pyproject.toml` | build and dependencies | §5.1 impl context | ~80 |
| `src/sdd_yolo/config.py` | dataclass config schema | §4/§5 | ~180 |
| `src/sdd_yolo/data.py` | YOLO detection dataset and collate | §3.1 | ~220 |
| `configs/default.toml` | baseline runtime config | §4-§5 | ~80 |
| `configs/paper.toml` | paper-oriented defaults | §4-§5 | ~80 |
| `configs/debug.toml` | quick smoke config | local | ~40 |
| `tests/test_config_and_data.py` | basic parser tests | — | ~120 |

## Test Plan
```bash
uv run pytest tests/test_config_and_data.py -v
```

## References
- Paper: Section 3 (dataset), Section 5.1 (implementation)
- Feeds into: PRD-02
