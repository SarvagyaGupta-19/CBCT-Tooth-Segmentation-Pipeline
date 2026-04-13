# CBCT Tooth Segmentation Pipeline — Dobbe AI

**Author:** Sarvagya | **Python 3.10+ · PyTorch 2.1 · nnU-Net v2**

---

## Pipeline Overview

```
Input CBCT (.mha/.nii/.nii.gz/DICOM)
    → Preprocessing (resample 0.4mm, HU clip, z-norm)
    → nnU-Net 3D full-res (ResEncL)
    → Post-processing (LCC per tooth class)
    → Outputs: mask.nii.gz + jaw_mask.nii.gz + labels.json
    → Viewer: viewer/index.html (self-contained)
```

**Two inference paths:**
| Path | Script | Expected Dice |
|---|---|---|
| Primary (nnU-Net v2) | `scripts/nnunet_pipeline.py` | ~0.91 |
| Fallback (custom U-Net) | `scripts/unet_training.py` + `scripts/unet_inference.py` | ~0.80 |

---

## Design Choices

**Backbone:** nnU-Net v2 3D full-res with ResEncL planner — auto-configures patch size, batch size, augmentation from dataset statistics. Published SOTA on dental CBCT benchmarks.

**Multi-task heads:** FDI segmentation (33 classes), jaw separation (3 classes), restoration detection (32 sigmoid logits per tooth — trained with BCE, fallback uses HU>1500 thresholding).

**Preprocessing:** Isotropic 0.4mm B-spline resampling → HU clip [-1000, 3000] → z-score normalisation → pad to 16-divisible.

**Inference:** Sliding-window with 96³ patches, 50% overlap, Gaussian importance weighting. Post-processing: largest connected component per tooth class.

---

## Dataset & Splits

**Primary:** [ToothFairy2](https://ditto.ing.unimore.it/toothfairy2) — multi-class CBCT tooth segmentation with FDI labels.

| Split | % | Purpose |
|---|---|---|
| Train | 70 | Training + augmentation |
| Val | 15 | Hyperparameter tuning |
| Test | 15 | Held-out evaluation only |

Deterministic split (seed=42). Split JSON files saved on first run.

---

## Quickstart

### Docker
```bash
docker build -t dobbe-cbct .

# Inference
docker run --gpus all -v /data:/data -v /runs:/runs -v /results:/results dobbe-cbct

# Training (nnU-Net)
docker run --gpus all -v /data:/data -v /runs:/runs dobbe-cbct \
  python3 scripts/nnunet_pipeline.py full \
    --toothfairy2-dir /data/toothfairy2 \
    --test-volume /data/demo/example.nii.gz \
    --epochs 300
```

### Without Docker
```bash
pip install -r requirements.txt

# nnU-Net path (recommended)
export nnUNet_raw=/data/nnunet_raw
export nnUNet_preprocessed=/data/nnunet_preprocessed
export nnUNet_results=/runs/nnunet_results

python3 scripts/nnunet_pipeline.py convert --toothfairy2-dir /data/toothfairy2
python3 scripts/nnunet_pipeline.py preprocess --dataset-id 100
python3 scripts/nnunet_pipeline.py train --dataset-id 100 --fold 0 --epochs 300
python3 scripts/nnunet_pipeline.py infer --input scan.nii.gz --output-dir results/

# Fallback path (custom U-Net)
python3 scripts/unet_preprocessing.py scan.nii.gz scan_pp.nii.gz
python3 scripts/unet_training.py --data-dir /data/toothfairy2 --output-dir runs/v1
python3 scripts/unet_inference.py --input scan.nii.gz --weights runs/v1/best_model.pth
```

### Google Colab (GPU training)
Upload data to Google Drive. Use the provided training notebook workflow:
```python
# Mount drive, install deps, set nnUNet env vars, run training
!pip install nnunetv2 SimpleITK medpy
!python scripts/nnunet_integration.py train --dataset-id 100 --fold 0 --epochs 300
```

### 3D Viewer
Open `viewer/index.html` directly in browser. Supports:
- Three synchronized slice views (axial/coronal/sagittal) with scroll navigation
- Click to reposition crosshair
- Window/level and opacity controls
- 3D point cloud rendering with orbit controls
- Drag-and-drop `labels.json` to load real predictions
- Keyboard: `T` toggle CT, `S` toggle seg, `3` toggle 3D, arrows for navigation

---

## Repository Structure

```
cbct_seg/
├── scripts/
│   ├── unet_preprocessing.py   # Volume I/O + resampling + normalisation
│   ├── unet_training.py        # Custom 3D U-Net (fallback)
│   ├── unet_inference.py       # Sliding-window inference + label JSON
│   └── nnunet_pipeline.py      # nnU-Net v2 full workflow
├── viewer/
│   └── index.html              # Self-contained 3D viewer
├── Dockerfile
├── requirements.txt
└── README.md
```

## Evaluation Metrics

| Metric | Description |
|---|---|
| Mean Dice (per tooth) | Voxel overlap, FDI 11–48 |
| Hausdorff95 | Surface boundary accuracy (mm) |
| Jaw accuracy | Mandible/maxilla classification |
| Restoration AUC | Per-tooth binary classification |
