# 3D CBCT Dental Segmentation Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c) ![nnU-Net](https://img.shields.io/badge/nnU--Net-v2-green)

A production-ready pipeline schema designed for 3-D maxillofacial CBCT tooth segmentation. The codebase is strictly structured to seamlessly process `.mha`/`.nii.gz` inputs to deliver high-accuracy FDI tooth numbering, jawbone classification, and metallic restoration detection.

> [!IMPORTANT]
> **Project Status: Architecture & Codebase Complete (Untrained)**
> *Due to physical hardware and GPU compute constraints, the deep-learning models detailed in this repository have not been actively trained. This repository currently serves as a complete structural, architectural, and procedural codebase mapping out exactly how the inference, preprocessing, and 3D UI logic should be executed once compute becomes available.*

<p align="center">
  <img src="./assets/viewer_screenshot.png" alt="3D CBCT HTML Viewer" width="85%"/>
</p>

## 🎯 System Architecture

This project is architected as an end-to-end medical vision pipeline, utilizing a two-pronged strategy for redundancy and performance scaling:

### 1. Primary Segmentation Engine (`nnU-Net v2`)
* **Backbone:** 3D full-resolution layout using a Residual Encoder (`ResEncL`) preset.
* **Why nnU-Net?:** Automatically analyzes voxel spacings and class imbalances to autonomously configure batch sizes and augmentations. It eliminates human bias from the preprocessing layer, achieving benchmark-breaking performance in medical datasets (like ToothFairy2).
* **Multi-Task Topology:** Designed to output 33 channels (Background + 32 discrete FDI teeth).

### 2. Algorithmic Post-Processing
* **Mask Isolation:** Runs explicit Largest-Connected-Component (LCC) filtering per tooth class to drop floating artifacts.
* **Jaw Anatomy Mapping:** Extracts independent Mandible & Maxilla masks by casting vertical location voting across isolated tooth coordinates.
* **Restoration Detection:** Employs empirical Hounsfield Unit (HU) thresholding (>1500 HU density checks) on isolated tooth crops to generate binary crown/implant presence flags.

### 3. GPU-Accelerated 3D Visualization
* **WebGL Interface:** A zero-dependency HTML viewer (`viewer/index.html`) engineered to ingest raw NIfTI scans and `.json` label graphs simultaneously. It overlaps transparent FDI semantic meshes dynamically on top of native CBCT densities in the browser.

---

## 📂 Codebase Structure

The repository is modularly segmented into pipeline actions:

```text
cbct_seg/
├── scripts/
│   ├── nnunet_pipeline.py      # Production workflow (Convert → Preprocess → Train → Infer)
│   ├── unet_inference.py       # Standalone Sliding-window inference logic
│   ├── unet_training.py        # Fallback PyTorch 3D U-Net baseline architecture
│   └── unet_preprocessing.py   # Baseline volume manipulation (Isotropic Resampling / HU Norm)
├── viewer/
│   └── index.html              # Custom interactive 3D WebGL viewer (Drag-and-drop .json)
├── assets/                     # Documentation UI materials
├── COLAB_GUIDE.md              # Explicit execution path mapping for Google Colab GPUs
└── Dockerfile                  # Baseline Ubuntu + CUDA Container spec orchestration
```

---

## 🚀 Future Execution (Google Colab / Docker)

When compute becomes available, the pipeline is fully pre-configured to build entirely in the cloud.

### Google Colab execution
1. Ensure the datatset `.zip` is uploaded to Google Drive.
2. Open `COLAB_GUIDE.md` and execute the 13 defined cells sequentially. 
3. The codebase dynamically interfaces with broken zip structures and establishes proper `dataset.json` keys to execute Epoch 1.

### Local Docker orchestration
```bash
# Build the container
docker build -t cbct-seg .

# Simulate single-scan inference (Requires pre-trained weights in /runs)
docker run --gpus all -v /data:/data -v /runs:/runs -v /results:/results cbct-seg
```

## 📊 Evaluation Criteria

Once trained, the pipeline will evaluate via the following metrics across 33 semantic classes:
* **Dice Coefficient:** Volumetric overlap accuracy per FDI tooth.
* **Hausdorff95:** Edge-boundary structural alignment tracking in absolute mm.
