# DEF-sdd-yolo Pipeline Map

1. **Input**: RGB image/frame (ground-to-air scene)
2. **Backbone**: CSP-style feature extraction
3. **Neck**: FPN/PAN fusion with added **P2 branch** (4x downsample)
4. **Attention**: spatial + channel gating to suppress clutter
5. **Head**: anchor-free, DFL-free decoupled detection head
6. **Train-time assignment**: one-to-many + one-to-one (hybrid)
7. **Infer-time assignment**: one-to-one (NMS-free mode)
8. **Outputs**: boxes, objectness, classes, optional latency/FPS telemetry
