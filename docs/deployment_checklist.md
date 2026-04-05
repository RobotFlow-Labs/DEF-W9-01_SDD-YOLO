# DEF-sdd-yolo Deployment Checklist

- [ ] Verify model checkpoint provenance.
- [ ] Confirm DroneSOD-30K split files are mounted.
- [ ] Run `python scripts/export_onnx.py` and validate output graph.
- [ ] Benchmark CPU and GPU FPS against target thresholds.
- [ ] Validate API `/health`, `/ready`, `/predict` in container.
- [ ] Validate NMS-free mode against integration tests.
- [ ] Document any quantization deltas for INT8 export.
