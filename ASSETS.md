# DEF-sdd-yolo — Asset Manifest

## Paper
- Title: SDD-YOLO: A Small-Target Detection Framework for Ground-to-Air Anti-UAV Surveillance with Edge-Efficient Deployment
- ArXiv: 2603.25218
- Authors: Pengyu Chen, Haotian Sa, Yiwei Hu, Yuhan Cheng, Junbo Wang
- PDF: `papers/2603.25218.pdf`

## Status: ALMOST
- Paper is available and parsed.
- No authoritative upstream repository or dataset download URL is provided in the paper/arXiv source.
- Local implementation scaffold is complete and runnable with YOLO-format data.

## Pretrained Weights
| Model | Size | Source | Path on Server | Status |
|---|---:|---|---|---|
| YOLO26n teacher/student init | N/A | Ultralytics (referenced by paper) | `/mnt/forge-data/models/yolo26n.pt` | UNKNOWN |
| YOLOv5n baseline | N/A | Ultralytics | `/mnt/forge-data/models/yolov5n.pt` | UNKNOWN |
| Optional YOLO11n fallback | N/A | shared infra | `/mnt/forge-data/models/yolo11n.pt` | AVAILABLE (shared infra map) |

## Datasets
| Dataset | Size | Split | Source | Path | Status |
|---|---:|---|---|---|---|
| DroneSOD-30K | ~30k images | train 30,655 / val 14,010 / test 3,085 | paper (no public URL in manuscript) | `/mnt/forge-data/datasets/dronesod30k/` | MISSING/UNKNOWN |
| Anti-UAV410 (cross-domain baseline) | sequence dataset | paper reference | public benchmark | `/mnt/forge-data/datasets/anti_uav410/` | UNKNOWN |
| Det-Fly / Drone-vs-Bird (optional) | varies | benchmark | public benchmark | `/mnt/forge-data/datasets/g2a_uav/` | UNKNOWN |

## Hyperparameters (from paper)
| Param | Value | Paper Section |
|---|---|---|
| Input resolution | 1024 x 1024 (architecture discussion) | 4.2 |
| Detection scales | P2/P3/P4/P5 with added P2 at 4x | 4.2 |
| DFL coefficient | 0.0 (DFL-free) | 4.3 |
| Regression loss | Wise-IoU v3 style IoU regression | Eq. (3), 4.3 |
| Inference assignment | one-to-one, NMS-free | 4.4 |
| Attention | dual spatial + channel attention | Eq. (4), 4.5 |
| Optimizer strategy | MuSGD hybrid (Muon-like + SGD for 1D params) | Eq. (5), 4.6 |
| Small-target assignment | STAL | 4.6 |
| Distillation temperature | 3.0 | Eq. (7), 4.7 |
| Distillation lambda | 0.5 | 4.7 |
| Data augmentation | Mosaic, Mixup, multi-scale | 5.1 |
| Epochs / LR / batch | not explicitly reported | N/A |

## Expected Metrics (paper)
| Benchmark | Metric | Paper Value | Our Target |
|---|---|---:|---:|
| DroneSOD-30K (SDD-YOLO-n final) | mAP@0.5 | 0.860 | >= 0.84 |
| DroneSOD-30K (SDD-YOLO-n final) | mAP@0.5:0.95 | 0.480 | >= 0.46 |
| DroneSOD-30K (SDD-YOLO-n final) | CPU FPS | 35.0 | >= 30 |
| DroneSOD-30K (SDD-YOLO-n final) | GPU FPS (RTX 5090) | 226.0 | >= 180 |

## Data Contracts Needed for Build
- YOLO-format labels (`class cx cy w h`) for DroneSOD-30K splits.
- Split files (`train.txt`, `val.txt`, `test.txt`) with image-relative paths.
- Optional teacher logits/checkpoints for feature distillation.
