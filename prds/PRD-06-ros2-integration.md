# PRD-06: ROS2 Integration for ANIMA Runtime

> Module: DEF-sdd-yolo | Priority: P1
> Depends on: PRD-05
> Status: ⬜ Not started

## Objective
Provide ROS2 integration layer so SDD-YOLO can run as an ANIMA node in robotics pipelines.

## Context (from paper)
Application target is real-time anti-UAV surveillance; ROS2 integration is required by ANIMA platform runtime.

## Acceptance Criteria
- [ ] ROS2 node wrapper implemented with clean process interface.
- [ ] Topic contracts documented.
- [ ] Stub launch file and module manifest present.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---:|
| `src/sdd_yolo/ros2/node.py` | ROS2 node adapter | ANIMA runtime | ~180 |
| `src/sdd_yolo/ros2/messages.py` | typed payload adapters | ANIMA runtime | ~90 |
| `anima_module.yaml` | module metadata contract | ANIMA runtime | ~80 |
| `docs/ros2_topics.md` | pub/sub interface docs | — | ~70 |

## Test Plan
```bash
uv run pytest -k ros2 -q
```

## References
- Platform requirement (ANIMA)
- Feeds into: PRD-07
