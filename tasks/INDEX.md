# DEF-sdd-yolo Task Index

## Build Order
| Task | Title | Depends | Status |
|---|---|---|---|
| PRD-0101 | Scaffold packaging and config dataclasses | None | ✅ |
| PRD-0102 | Implement YOLO-format dataset ingestion | PRD-0101 | ✅ |
| PRD-0103 | Add config and data tests | PRD-0102 | ✅ |
| PRD-0201 | Build backbone and neck blocks | PRD-0103 | ✅ |
| PRD-0202 | Add dual attention and P2 branch | PRD-0201 | ✅ |
| PRD-0203 | Implement DFL-free detection head | PRD-0202 | ✅ |
| PRD-0204 | Implement training losses with STAL weighting | PRD-0203 | ✅ |
| PRD-0205 | Assemble full SDDYOLO model | PRD-0204 | ✅ |
| PRD-0206 | Add model-forward unit tests | PRD-0205 | ✅ |
| PRD-0301 | Build inference runner core | PRD-0206 | ✅ |
| PRD-0302 | Add visualization output writer | PRD-0301 | ✅ |
| PRD-0303 | Ship inference CLI and smoke tests | PRD-0302 | ✅ |
| PRD-0401 | Implement IoU/AP metric utilities | PRD-0303 | ✅ |
| PRD-0402 | Implement evaluation CLI and report output | PRD-0401 | ✅ |
| PRD-0501 | Implement FastAPI service endpoints | PRD-0303 | ✅ |
| PRD-0502 | Add Docker serving artifacts | PRD-0501 | ✅ |
| PRD-0503 | Add API tests | PRD-0502 | ✅ |
| PRD-0601 | Add ROS2 bridge stubs and manifest | PRD-0503 | ✅ |
| PRD-0701 | Implement ONNX export helper and CLI | PRD-0402 | ✅ |
| PRD-0702 | Write deployment checklist and report templates | PRD-0701 | ✅ |
