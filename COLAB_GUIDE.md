# Google Colab — Training Guide (ToothFairy2 .mha format)

## Dataset Facts
- Files are `.mha` (not .nii.gz) — nnU-Net handles this natively via SimpleITKIO
- Images: `<caseID>_0000.mha`, Labels: `<caseID>.mha`
- Label IDs = FDI numbers directly (11-48) + anatomy (1-10)
- Labels 19, 20, 29, 30, 39 don't exist. Label 40 = "NA"

---

## Cell 1 — GPU check
```python
!nvidia-smi
import torch
print(f"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}")
```

## Cell 2 — Install deps
```python
!pip install -q nnunetv2 SimpleITK nibabel medpy scipy
```

## Cell 3 — Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Cell 4 — Unzip subset from Drive
```python
# If you uploaded a zip:
!cp "/content/drive/MyDrive/toothfairy2_subset.zip" /content/
!unzip -q /content/toothfairy2_subset.zip -d /content/toothfairy2_subset
!rm /content/toothfairy2_subset.zip

# Note: We don't need to manually verify the path here anymore.
# Cell 6 will automatically find the extracted folder for you!
```

## Cell 5 — Set nnU-Net env vars
```python
import os

os.environ['nnUNet_raw'] = '/content/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '/content/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '/content/drive/MyDrive/nnunet_results'

!mkdir -p /content/nnUNet_raw /content/nnUNet_preprocessed
!mkdir -p /content/drive/MyDrive/nnunet_results
```

## Cell 6 — Link dataset into nnU-Net folder
```python
import os, glob, shlex

# Automatically find where the dataset actually extracted
search_results = glob.glob('/content/**/imagesTr', recursive=True)
if not search_results:
    raise FileNotFoundError("Could not find 'imagesTr'. Did the unzip fail?")

actual_dataset_dir = os.path.dirname(search_results[0])
print(f"Dataset found at: {actual_dataset_dir}")

DATASET_ID = 100
DATASET_NAME = f"Dataset{DATASET_ID:03d}_ToothFairy2"
RAW_PATH = f"/content/nnUNet_raw/{DATASET_NAME}"
os.makedirs(RAW_PATH, exist_ok=True)

# Safely create symlink (handles spaces like 'ToothFairy 2')
safe_src = shlex.quote(actual_dataset_dir)
!ln -sTf {safe_src} {RAW_PATH}

# Count files
imgs = glob.glob(f"{RAW_PATH}/imagesTr/*_0000.mha")
lbls = glob.glob(f"{RAW_PATH}/labelsTr/*.mha")
print(f"Images: {len(imgs)}, Labels: {len(lbls)}")
```

## Cell 7 — Fix dataset.json
```python
import json

ds_path = f"{RAW_PATH}/dataset.json"
with open(ds_path) as f:
    ds = json.load(f)

# Fix key name if wrong (original has "channels_names", nnU-Net needs "channel_names")
if "channels_names" in ds and "channel_names" not in ds:
    ds["channel_names"] = ds.pop("channels_names")

# Fix numTraining
imgs = glob.glob(f"{RAW_PATH}/imagesTr/*_0000.mha")
ds["numTraining"] = len(imgs)

# Ensure SimpleITKIO for .mha
ds["overwrite_image_reader_writer"] = "SimpleITKIO"

# Fix background key (nnU-Net v2 needs lowercase "background")
if "Background" in ds.get("labels", {}):
    ds["labels"]["background"] = ds["labels"].pop("Background")

with open(ds_path, "w") as f:
    json.dump(ds, f, indent=2)

print(f"Fixed dataset.json: {ds['numTraining']} cases, file_ending={ds['file_ending']}")
print(f"Labels: {len(ds['labels'])} classes")
```

## Cell 8 — Plan and preprocess
```python
# ~5-10 min for 80 cases
!nnUNetv2_plan_and_preprocess -d {DATASET_ID} --verify_dataset_integrity -c 3d_fullres
```

## Cell 9 — Train
```python
# ~4-6 hours on T4 for 80 cases
# Results saved to Drive (persistent across disconnects)
!nnUNetv2_train {DATASET_ID} 3d_fullres 0 -tr nnUNetTrainer --npz
```

**If Colab disconnects:** Re-run Cells 2, 3, 4, 5, 6, then Cell 9 again. nnU-Net auto-resumes.

## Cell 10 — Inference on one volume
```python
import shutil, glob

# Pick first case
test_img = sorted(glob.glob(f"{RAW_PATH}/imagesTr/*_0000.mha"))[0]
print(f"Running inference on: {test_img}")

INPUT_DIR = "/content/infer_input"
OUTPUT_DIR = "/content/results"
!mkdir -p {INPUT_DIR} {OUTPUT_DIR}
!cp {test_img} {INPUT_DIR}/

!nnUNetv2_predict \
    -i {INPUT_DIR} \
    -o {OUTPUT_DIR} \
    -d {DATASET_ID} \
    -c 3d_fullres \
    -tr nnUNetTrainer \
    -f 0

!ls -la {OUTPUT_DIR}/
```

## Cell 11 — Generate labels.json
```python
import SimpleITK as sitk
import numpy as np
import json, os

FDI_NAMES = {
    11:'Upper Right Central Incisor', 12:'Upper Right Lateral Incisor',
    13:'Upper Right Canine', 14:'Upper Right First Premolar',
    15:'Upper Right Second Premolar', 16:'Upper Right First Molar',
    17:'Upper Right Second Molar', 18:'Upper Right Third Molar',
    21:'Upper Left Central Incisor', 22:'Upper Left Lateral Incisor',
    23:'Upper Left Canine', 24:'Upper Left First Premolar',
    25:'Upper Left Second Premolar', 26:'Upper Left First Molar',
    27:'Upper Left Second Molar', 28:'Upper Left Third Molar',
    31:'Lower Left Central Incisor', 32:'Lower Left Lateral Incisor',
    33:'Lower Left Canine', 34:'Lower Left First Premolar',
    35:'Lower Left Second Premolar', 36:'Lower Left First Molar',
    37:'Lower Left Second Molar', 38:'Lower Left Third Molar',
    41:'Lower Right Central Incisor', 42:'Lower Right Lateral Incisor',
    43:'Lower Right Canine', 44:'Lower Right First Premolar',
    45:'Lower Right Second Premolar', 46:'Lower Right First Molar',
    47:'Lower Right Second Molar', 48:'Lower Right Third Molar',
}
TOOTH_FDIS = [fdi for fdi in range(11, 49) if fdi not in (19, 20, 29, 30, 39, 40)]

# Find prediction file
pred_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.mha') or f.endswith('.nii.gz')]
pred_path = os.path.join(OUTPUT_DIR, pred_files[0])
print(f"Prediction: {pred_path}")

pred_img = sitk.ReadImage(pred_path)
pred_arr = sitk.GetArrayFromImage(pred_img)
scan_img = sitk.ReadImage(test_img)
scan_arr = sitk.GetArrayFromImage(scan_img).astype(np.float32)

# Jaw mask
jaw_arr = np.zeros_like(pred_arr, dtype=np.uint8)
jaw_arr[pred_arr == 1] = 1  # mandible
jaw_arr[pred_arr == 2] = 2  # maxilla
for fdi in range(11, 29):
    jaw_arr[pred_arr == fdi] = 2
for fdi in range(31, 49):
    jaw_arr[pred_arr == fdi] = 1

jaw_img = sitk.GetImageFromArray(jaw_arr)
jaw_img.CopyInformation(pred_img)
sitk.WriteImage(jaw_img, os.path.join(OUTPUT_DIR, 'jaw_mask.nii.gz'), True)

# Rename to mask.nii.gz
mask_out = os.path.join(OUTPUT_DIR, 'mask.nii.gz')
if not os.path.exists(mask_out):
    # Convert .mha prediction to .nii.gz
    sitk.WriteImage(pred_img, mask_out, True)

# Per-tooth labels
teeth = []
for fdi in TOOTH_FDIS:
    mask = pred_arr == fdi
    if not mask.any():
        continue
    hu_vals = scan_arr[mask]
    rest_prob = float(np.mean(hu_vals > 1500))
    jaw = 'maxilla' if fdi < 31 else 'mandible'
    teeth.append({
        'fdi_id': int(fdi),
        'fdi_name': FDI_NAMES.get(fdi, 'Unknown'),
        'class_index': int(fdi),
        'jaw': jaw,
        'voxel_count': int(mask.sum()),
        'restoration_prob': round(rest_prob, 4),
        'restoration_flag': rest_prob > 0.15
    })

labels = {'num_teeth': len(teeth), 'teeth': teeth}
with open(os.path.join(OUTPUT_DIR, 'labels.json'), 'w') as f:
    json.dump(labels, f, indent=2)

print(f"\nDetected {len(teeth)} teeth:")
for t in teeth:
    flag = 'R' if t['restoration_flag'] else 'H'
    print(f"  FDI {t['fdi_id']:2d} | {t['fdi_name']:35s} | {t['jaw']:9s} | {t['voxel_count']:6d} vox | {flag}")
```

## Cell 12 — Save to Drive
```python
!mkdir -p /content/drive/MyDrive/cbct_results/
!cp /content/results/mask.nii.gz /content/drive/MyDrive/cbct_results/
!cp /content/results/jaw_mask.nii.gz /content/drive/MyDrive/cbct_results/
!cp /content/results/labels.json /content/drive/MyDrive/cbct_results/
print("Done! Download labels.json and drag into viewer/index.html")
```

## Cell 13 — Evaluate
```python
from medpy.metric.binary import dc, hd95

# Get matching ground truth
case_name = os.path.basename(test_img).replace("_0000.mha", "")
gt_path = f"{RAW_PATH}/labelsTr/{case_name}.mha"
gt_arr = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
spacing = pred_img.GetSpacing()[::-1]

dices = []
for fdi in TOOTH_FDIS:
    pred_c = pred_arr == fdi
    gt_c = gt_arr == fdi
    if not gt_c.any():
        continue
    d = dc(pred_c, gt_c) if pred_c.any() else 0.0
    dices.append(d)
    print(f"  FDI {fdi}: Dice = {d:.4f}")

print(f"\nMean Dice: {np.mean(dices):.4f} over {len(dices)} teeth")
```
