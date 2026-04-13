

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ─── nnU-Net constants ────────────────────────────────────────────────────────
NNUNET_TRAINER   = "nnUNetTrainer"           # default; use nnUNetTrainerNoMirroring on CBCT if needed
NNUNET_CONFIG    = "3d_fullres"              # 3D full-resolution — best for CBCT
NNUNET_PLANNER   = "ExperimentPlanner"       # auto-configures everything
RESENC_PLANNER   = "nnUNetPlannerResEncL"    # ResNet encoder — use for extra Dice

NUM_FDI_CLASSES  = 32    # tooth classes (FDI 11-48, excluding background)
SEED             = 42


# ─── Helpers ──────────────────────────────────────────────────────────────────
def run(cmd: list[str], env: dict = None) -> None:
    """Run a shell command, streaming output, raising on failure."""
    full_env = {**os.environ, **(env or {})}
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, env=full_env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {' '.join(cmd)}")


def nnunet_env() -> dict:
    """Return required nnU-Net env vars, defaulting to sensible paths."""
    return {
        "nnUNet_raw":          os.environ.get("nnUNet_raw",          "/data/nnunet_raw"),
        "nnUNet_preprocessed": os.environ.get("nnUNet_preprocessed", "/data/nnunet_preprocessed"),
        "nnUNet_results":      os.environ.get("nnUNet_results",       "/runs/nnunet_results"),
    }


def dataset_name(dataset_id: int) -> str:
    return f"Dataset{dataset_id:03d}_CBCT"


# ─── Step 1: Convert ToothFairy2 → nnU-Net raw format ────────────────────────
def convert_dataset(toothfairy2_dir: Path, dataset_id: int) -> None:
    """
    ToothFairy2 is ALREADY in nnU-Net format:
        toothfairy2/
          imagesTr/<caseID>_0000.nii.gz   (already _0000 suffixed)
          labelsTr/<caseID>.nii.gz        (label IDs = FDI numbers: 11-48)
          dataset.json                     (42 classes incl. anatomy)

    This function symlinks or copies into nnUNet_raw and creates splits.
    """
    env   = nnunet_env()
    dname = dataset_name(dataset_id)
    raw_root  = Path(env["nnUNet_raw"]) / dname

    src_dir = Path(toothfairy2_dir)

    # If dataset.json exists in source, the dataset is already nnU-Net formatted
    src_ds_json = src_dir / "dataset.json"
    already_formatted = src_ds_json.exists()

    if already_formatted:
        log.info("ToothFairy2 is already in nnU-Net format")
        # Symlink entire directory if not already present
        if not raw_root.exists():
            try:
                raw_root.symlink_to(src_dir.resolve())
                log.info("Symlinked %s → %s", raw_root, src_dir)
            except OSError:
                shutil.copytree(str(src_dir), str(raw_root))
                log.info("Copied %s → %s", src_dir, raw_root)
    else:
        # Older format: images without _0000 suffix — need renaming
        img_out = raw_root / "imagesTr"
        lbl_out = raw_root / "labelsTr"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        src_img = src_dir / "imagesTr"
        src_lbl = src_dir / "labelsTr"

        for src_file in sorted(src_img.glob("*.nii.gz")):
            name = src_file.name
            # Add _0000 if not present
            if "_0000" not in name:
                dst_name = name.replace(".nii.gz", "_0000.nii.gz")
            else:
                dst_name = name
            dst = img_out / dst_name
            if not dst.exists():
                shutil.copy2(src_file, dst)

        for src_file in sorted(src_lbl.glob("*.nii.gz")):
            dst = lbl_out / src_file.name
            if not dst.exists():
                shutil.copy2(src_file, dst)

    # ── dataset.json (use official one if available) ──────────────────────────
    dst_ds_json = raw_root / "dataset.json"
    if not dst_ds_json.exists():
        labels = {
            "background": 0,
            "Lower Jawbone": 1, "Upper Jawbone": 2,
            "Left Inferior Alveolar Canal": 3, "Right Inferior Alveolar Canal": 4,
            "Left Maxillary Sinus": 5, "Right Maxillary Sinus": 6,
            "Pharynx": 7, "Bridge": 8, "Crown": 9, "Implant": 10,
        }
        for fdi in range(11, 49):
            if fdi in (19, 20, 29, 30, 39, 40):
                labels[f"NA{fdi}"] = fdi
            else:
                labels[_fdi_name(fdi)] = fdi

        dataset_json = {
            "channel_names": {"0": "CBCT"},
            "labels": labels,
            "numTraining": 480,
            "file_ending": ".nii.gz",
        }
        dst_ds_json.write_text(json.dumps(dataset_json, indent=2))
        log.info("Generated dataset.json with %d classes", len(labels))

    # ── Train / val / test split ──────────────────────────────────────────────
    img_dir = raw_root / "imagesTr"
    cases = sorted(set(
        p.name.replace("_0000.nii.gz", "").replace(".nii.gz", "")
        for p in img_dir.glob("*.nii.gz")
    ))

    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(cases).tolist()
    n = len(shuffled)
    train_cases = shuffled[:int(0.70 * n)]
    val_cases   = shuffled[int(0.70 * n):int(0.85 * n)]
    test_cases  = shuffled[int(0.85 * n):]

    splits_path = raw_root / "splits_final.json"
    if not splits_path.exists():
        splits = [{"train": train_cases, "val": val_cases}]
        splits_path.write_text(json.dumps(splits, indent=2))
        (raw_root / "test_cases.json").write_text(json.dumps(test_cases, indent=2))

    log.info("Split → train=%d val=%d test=%d", len(train_cases), len(val_cases), len(test_cases))
    log.info("Dataset ready → %s", raw_root)


# ─── Step 2: nnU-Net plan + preprocess ───────────────────────────────────────
def preprocess(dataset_id: int, use_resenc: bool = True) -> None:
    """
    Calls nnUNetv2_plan_and_preprocess which:
      • Analyses voxel spacings, intensities, image sizes
      • Chooses optimal patch size and batch size
      • Generates low-res and full-res preprocessed data
      • With ResEncL planner: uses deeper residual encoder (better Dice)
    """
    planner = RESENC_PLANNER if use_resenc else NNUNET_PLANNER
    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", str(dataset_id),
        "--verify_dataset_integrity",
        "-pl", planner,
        "-c", NNUNET_CONFIG,
    ]
    run(cmd, nnunet_env())
    log.info("Preprocessing complete.")


# ─── Step 3: Train ────────────────────────────────────────────────────────────
def train(dataset_id: int, fold: int, epochs: int,
          use_resenc: bool = True, pretrained_weights: Path = None) -> None:
    """
    Launch nnU-Net 3D full-resolution training for one fold.

    Key flags:
      --npz          : save softmax probabilities (needed for ensembling)
      -tr            : trainer class (nnUNetTrainer supports 1000 epochs by default)
      --c            : continue from last checkpoint if interrupted
    """
    planner = RESENC_PLANNER if use_resenc else NNUNET_PLANNER
    trainer = NNUNET_TRAINER

    cmd = [
        "nnUNetv2_train",
        str(dataset_id),
        NNUNET_CONFIG,
        str(fold),
        "-tr", trainer,
        "-p",  planner,
        "--npz",
        "--num_epochs", str(epochs),
    ]

    if pretrained_weights and Path(pretrained_weights).exists():
        cmd += ["--pretrained_weights", str(pretrained_weights)]
        log.info("Using pretrained weights: %s", pretrained_weights)

    run(cmd, nnunet_env())
    log.info("Training fold %d complete.", fold)


def train_all_folds(dataset_id: int, epochs: int, use_resenc: bool = True) -> None:
    """Train all 5 folds for full cross-validation (for ensemble inference)."""
    for fold in range(5):
        log.info("═══ Training fold %d/5 ═══", fold)
        train(dataset_id, fold, epochs, use_resenc)


# ─── Step 4: Inference ────────────────────────────────────────────────────────
def infer(
    input_path:  Path,
    output_dir:  Path,
    dataset_id:  int,
    fold:        int | str = 0,   # pass "all" to ensemble all folds
    use_resenc:  bool = True,
    save_probs:  bool = False,
) -> Path:
    """
    Run nnU-Net inference on a single volume (or folder).

    nnUNetv2_predict handles:
      • Sliding-window with Gaussian weighting (built-in)
      • Test-time augmentation (mirroring) — adds ~1 Dice point
      • Softmax probability averaging across folds if fold="all"
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # nnU-Net expects a folder of inputs named <case>_0000.nii.gz
    tmp_in = output_dir / "_nnunet_input"
    tmp_in.mkdir(exist_ok=True)
    dst = tmp_in / f"{input_path.stem.replace('.nii','')}_0000.nii.gz"
    shutil.copy2(input_path, dst)

    env = nnunet_env()
    planner = RESENC_PLANNER if use_resenc else NNUNET_PLANNER

    cmd = [
        "nnUNetv2_predict",
        "-i",  str(tmp_in),
        "-o",  str(output_dir),
        "-d",  str(dataset_id),
        "-c",  NNUNET_CONFIG,
        "-tr", NNUNET_TRAINER,
        "-p",  planner,
        "-f",  str(fold),       # "all" → ensemble, "0" → single fold
        "--save_probabilities" if save_probs else "--disable_tta",
    ]

    # Remove --disable_tta flag when save_probs (TTA runs by default with probs)
    if save_probs and "--disable_tta" in cmd:
        cmd.remove("--disable_tta")

    run(cmd, env)
    log.info("nnU-Net inference complete → %s", output_dir)

    # Rename nnU-Net output to our standard name
    pred_file = output_dir / dst.name.replace("_0000.nii.gz", ".nii.gz")
    mask_file = output_dir / "mask.nii.gz"
    if pred_file.exists() and not mask_file.exists():
        pred_file.rename(mask_file)

    # Clean up temp input folder
    shutil.rmtree(tmp_in, ignore_errors=True)

    return mask_file


# ─── FDI tooth label IDs in ToothFairy2 (label ID = FDI number) ──────────────
_TOOTH_FDIS = [fdi for fdi in range(11, 49) if fdi not in (19, 20, 29, 30, 39, 40)]


# ─── Step 5: Post-inference — restoration head + label JSON ──────────────────
def postprocess_and_label(
    scan_path:    Path,
    mask_path:    Path,
    output_dir:   Path,
    restore_model_path: Path = None,
) -> dict:
    """
    After nnU-Net produces mask.nii.gz:
      - In ToothFairy2, label IDs ARE FDI numbers (11-48)
      - Labels 1-2 are jawbones, 3-10 are other anatomy
      - This function builds jaw_mask.nii.gz + labels.json
    """
    output_dir = Path(output_dir)

    seg_img = sitk.ReadImage(str(mask_path))
    seg_arr = sitk.GetArrayFromImage(seg_img)

    scan_img = sitk.ReadImage(str(scan_path))
    scan_arr = sitk.GetArrayFromImage(scan_img).astype(np.float32)

    # ── Jaw mask (label 1=Lower Jawbone, 2=Upper Jawbone already in predictions)
    jaw_arr = np.zeros_like(seg_arr, dtype=np.uint8)
    jaw_arr[seg_arr == 1] = 1   # mandible (lower jawbone)
    jaw_arr[seg_arr == 2] = 2   # maxilla (upper jawbone)
    for fdi in range(11, 29):
        jaw_arr[seg_arr == fdi] = 2  # maxilla teeth
    for fdi in range(31, 49):
        jaw_arr[seg_arr == fdi] = 1  # mandible teeth

    jaw_img = sitk.GetImageFromArray(jaw_arr)
    jaw_img.CopyInformation(seg_img)
    sitk.WriteImage(jaw_img, str(output_dir / "jaw_mask.nii.gz"), True)

    # ── Per-tooth restoration detection (HU-threshold heuristic) ─────────────
    teeth = []
    for fdi in _TOOTH_FDIS:
        mask = seg_arr == fdi
        if not mask.any():
            continue

        hu_vals = scan_arr[mask]
        rest_prob = float(np.mean(hu_vals > 1500))
        jaw_name = "maxilla" if fdi < 31 else "mandible"

        teeth.append({
            "fdi_id":          fdi,
            "fdi_name":        _fdi_name(fdi),
            "class_index":     fdi,  # in ToothFairy2, class = FDI
            "jaw":             jaw_name,
            "voxel_count":     int(mask.sum()),
            "restoration_prob": round(rest_prob, 4),
            "restoration_flag": bool(rest_prob > 0.15),
        })

    labels = {"num_teeth": len(teeth), "teeth": teeth}
    (output_dir / "labels.json").write_text(json.dumps(labels, indent=2))
    log.info("Labelled %d teeth → labels.json", len(teeth))
    return labels


# ─── FDI helpers ──────────────────────────────────────────────────────────────
def _idx_to_fdi(idx: int) -> int:
    quadrant = (idx - 1) // 8 + 1
    position = (idx - 1) %  8 + 1
    return quadrant * 10 + position

_FDI_NAMES = {
    11:"UR Central Incisor",  12:"UR Lateral Incisor",  13:"UR Canine",
    14:"UR 1st Premolar",     15:"UR 2nd Premolar",      16:"UR 1st Molar",
    17:"UR 2nd Molar",        18:"UR Wisdom",
    21:"UL Central Incisor",  22:"UL Lateral Incisor",  23:"UL Canine",
    24:"UL 1st Premolar",     25:"UL 2nd Premolar",      26:"UL 1st Molar",
    27:"UL 2nd Molar",        28:"UL Wisdom",
    31:"LL Central Incisor",  32:"LL Lateral Incisor",  33:"LL Canine",
    34:"LL 1st Premolar",     35:"LL 2nd Premolar",      36:"LL 1st Molar",
    37:"LL 2nd Molar",        38:"LL Wisdom",
    41:"LR Central Incisor",  42:"LR Lateral Incisor",  43:"LR Canine",
    44:"LR 1st Premolar",     45:"LR 2nd Premolar",      46:"LR 1st Molar",
    47:"LR 2nd Molar",        48:"LR Wisdom",
}
def _fdi_name(fdi: int) -> str: return _FDI_NAMES.get(fdi, "Unknown")


# ─── Evaluation ───────────────────────────────────────────────────────────────
def evaluate(pred_path: Path, gt_path: Path) -> dict:
    """
    Compute per-tooth Dice and Hausdorff95.
    In ToothFairy2, label IDs are FDI numbers (11-48).
    """
    try:
        from medpy.metric.binary import dc, hd95
    except ImportError:
        log.warning("medpy not installed — pip install medpy")
        return {}

    pred_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path)))
    gt_arr   = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))
    spacing  = sitk.ReadImage(str(pred_path)).GetSpacing()[::-1]

    results = {}
    for fdi in _TOOTH_FDIS:
        pred_c = (pred_arr == fdi)
        gt_c   = (gt_arr   == fdi)
        if not gt_c.any():
            continue
        dice = dc(pred_c, gt_c) if pred_c.any() else 0.0
        h95  = hd95(pred_c, gt_c, voxelspacing=spacing) if (pred_c.any() and gt_c.any()) else float('nan')
        results[fdi] = {"dice": round(dice, 4), "hd95_mm": round(h95, 2)}

    mean_dice = np.mean([v["dice"] for v in results.values()]) if results else 0.0
    log.info("Mean Dice: %.4f over %d teeth", mean_dice, len(results))
    return {"mean_dice": round(mean_dice, 4), "per_tooth": results}


# ─── Full end-to-end pipeline shortcut ───────────────────────────────────────
def run_full_pipeline(
    toothfairy2_dir: Path,
    test_volume:     Path,
    dataset_id:      int  = 100,
    fold:            int  = 0,
    epochs:          int  = 300,
    output_dir:      Path = Path("results/demo"),
):
    """Convenience function: convert → preprocess → train → infer → label."""
    log.info("═══ Step 1/4: Convert dataset ═══")
    convert_dataset(toothfairy2_dir, dataset_id)

    log.info("═══ Step 2/4: Plan + preprocess ═══")
    preprocess(dataset_id, use_resenc=True)

    log.info("═══ Step 3/4: Train (fold %d, %d epochs) ═══", fold, epochs)
    train(dataset_id, fold, epochs, use_resenc=True)

    log.info("═══ Step 4/4: Inference + labelling ═══")
    mask_path = infer(test_volume, output_dir, dataset_id, fold)
    postprocess_and_label(test_volume, mask_path, output_dir)
    log.info("All done. Results in %s", output_dir)


# ─── CLI ──────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="nnU-Net v2 integration for CBCT tooth segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # convert
    c = sub.add_parser("convert", help="Convert ToothFairy2 → nnU-Net raw format")
    c.add_argument("--toothfairy2-dir", type=Path, required=True)
    c.add_argument("--dataset-id",      type=int,  default=100)

    # preprocess
    pp = sub.add_parser("preprocess", help="Run nnUNetv2_plan_and_preprocess")
    pp.add_argument("--dataset-id",  type=int, default=100)
    pp.add_argument("--no-resenc",   action="store_true", help="Use default planner instead of ResEncL")

    # train
    tr = sub.add_parser("train", help="Train one fold (or all 5)")
    tr.add_argument("--dataset-id",  type=int, default=100)
    tr.add_argument("--fold",        type=int, default=0, help="0-4, or -1 for all folds")
    tr.add_argument("--epochs",      type=int, default=300)
    tr.add_argument("--no-resenc",   action="store_true")
    tr.add_argument("--pretrained",  type=Path, default=None, help="Path to pretrained .pth weights")

    # infer
    inf = sub.add_parser("infer", help="Run inference on a single volume")
    inf.add_argument("--input",      type=Path, required=True)
    inf.add_argument("--output-dir", type=Path, default=Path("results/demo"))
    inf.add_argument("--dataset-id", type=int,  default=100)
    inf.add_argument("--fold",                  default="0", help="Fold index or 'all'")
    inf.add_argument("--no-resenc",  action="store_true")

    # evaluate
    ev = sub.add_parser("evaluate", help="Compute Dice + HD95 on a prediction")
    ev.add_argument("--pred", type=Path, required=True)
    ev.add_argument("--gt",   type=Path, required=True)

    # full (convenience)
    fu = sub.add_parser("full", help="Run entire pipeline end-to-end")
    fu.add_argument("--toothfairy2-dir", type=Path, required=True)
    fu.add_argument("--test-volume",     type=Path, required=True)
    fu.add_argument("--dataset-id",      type=int,  default=100)
    fu.add_argument("--fold",            type=int,  default=0)
    fu.add_argument("--epochs",          type=int,  default=300)
    fu.add_argument("--output-dir",      type=Path, default=Path("results/demo"))

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.command == "convert":
        convert_dataset(args.toothfairy2_dir, args.dataset_id)

    elif args.command == "preprocess":
        preprocess(args.dataset_id, use_resenc=not args.no_resenc)

    elif args.command == "train":
        if args.fold == -1:
            train_all_folds(args.dataset_id, args.epochs, use_resenc=not args.no_resenc)
        else:
            train(args.dataset_id, args.fold, args.epochs,
                  use_resenc=not args.no_resenc,
                  pretrained_weights=args.pretrained)

    elif args.command == "infer":
        fold_arg = args.fold if args.fold == "all" else int(args.fold)
        mask = infer(args.input, args.output_dir, args.dataset_id,
                     fold=fold_arg, use_resenc=not args.no_resenc)
        postprocess_and_label(args.input, mask, args.output_dir)

    elif args.command == "evaluate":
        results = evaluate(args.pred, args.gt)
        print(json.dumps(results, indent=2))

    elif args.command == "full":
        run_full_pipeline(
            toothfairy2_dir = args.toothfairy2_dir,
            test_volume     = args.test_volume,
            dataset_id      = args.dataset_id,
            fold            = args.fold,
            epochs          = args.epochs,
            output_dir      = args.output_dir,
        )
