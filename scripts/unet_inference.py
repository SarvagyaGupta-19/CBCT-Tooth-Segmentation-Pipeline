

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ─── FDI mapping ──────────────────────────────────────────────────────────────
def idx_to_fdi(idx: int) -> int:
    """Convert 1-based class index (1-32) to FDI notation (11-48)."""
    quadrant = (idx - 1) // 8 + 1       # 1-4
    position = (idx - 1) %  8 + 1       # 1-8
    return quadrant * 10 + position


FDI_NAMES = {
    11: "Upper Right Central Incisor",  12: "Upper Right Lateral Incisor",
    13: "Upper Right Canine",           14: "Upper Right 1st Premolar",
    15: "Upper Right 2nd Premolar",     16: "Upper Right 1st Molar",
    17: "Upper Right 2nd Molar",        18: "Upper Right 3rd Molar",
    21: "Upper Left Central Incisor",   22: "Upper Left Lateral Incisor",
    23: "Upper Left Canine",            24: "Upper Left 1st Premolar",
    25: "Upper Left 2nd Premolar",      26: "Upper Left 1st Molar",
    27: "Upper Left 2nd Molar",         28: "Upper Left 3rd Molar",
    31: "Lower Left Central Incisor",   32: "Lower Left Lateral Incisor",
    33: "Lower Left Canine",            34: "Lower Left 1st Premolar",
    35: "Lower Left 2nd Premolar",      36: "Lower Left 1st Molar",
    37: "Lower Left 2nd Molar",         38: "Lower Left 3rd Molar",
    41: "Lower Right Central Incisor",  42: "Lower Right Lateral Incisor",
    43: "Lower Right Canine",           44: "Lower Right 1st Premolar",
    45: "Lower Right 2nd Premolar",     46: "Lower Right 1st Molar",
    47: "Lower Right 2nd Molar",        48: "Lower Right 3rd Molar",
}

JAW_NAMES = {0: "background", 1: "mandible", 2: "maxilla"}


# ─── Sliding-window inference ─────────────────────────────────────────────────
def sliding_window_inference(
    model: torch.nn.Module,
    volume: torch.Tensor,          # (1, 1, D, H, W)
    patch_size: tuple = (96, 96, 96),
    overlap: float = 0.5,
    device: torch.device = torch.device("cpu"),
):
    """
    Run inference with overlapping patches and Gaussian importance weighting.
    Returns (logits_fdi, logits_jaw, logits_rest) without gradients.
    """
    model.eval()
    _, _, D, H, W = volume.shape
    pd, ph, pw = patch_size
    stride = tuple(int(p * (1 - overlap)) for p in patch_size)

    n_fdi = 33
    n_jaw = 3
    acc_fdi = torch.zeros(1, n_fdi, D, H, W, device=device)
    acc_jaw = torch.zeros(1, n_jaw, D, H, W, device=device)
    count   = torch.zeros(1, 1, D, H, W, device=device)

    # Gaussian weight map to downweight patch edges
    def gauss_map(size):
        coords = [torch.linspace(-1, 1, s) for s in size]
        mesh   = torch.meshgrid(*coords, indexing="ij")
        g = torch.exp(-0.5 * sum(m**2 for m in mesh) / 0.5**2)
        return g.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

    gmap = gauss_map(patch_size).to(device)

    # Generate patch start indices
    def starts(total, patch, step):
        s = list(range(0, total - patch, step))
        s.append(total - patch)
        return sorted(set(max(0, x) for x in s))

    rest_logits_list = []

    with torch.no_grad():
        for d0 in starts(D, pd, stride[0]):
            for h0 in starts(H, ph, stride[1]):
                for w0 in starts(W, pw, stride[2]):
                    patch = volume[:, :, d0:d0+pd, h0:h0+ph, w0:w0+pw].to(device)

                    # Pad if edge patch is smaller
                    pad = [0, max(0, pw - patch.shape[4]),
                           0, max(0, ph - patch.shape[3]),
                           0, max(0, pd - patch.shape[2])]
                    if any(p > 0 for p in pad):
                        patch = F.pad(patch, pad)

                    lf, lj, lr = model(patch)

                    # Crop back if padded
                    lf = lf[:, :, :min(pd, D-d0), :min(ph, H-h0), :min(pw, W-w0)]
                    lj = lj[:, :, :min(pd, D-d0), :min(ph, H-h0), :min(pw, W-w0)]
                    slc = np.s_[:, :, d0:d0+lf.shape[2], h0:h0+lf.shape[3], w0:w0+lf.shape[4]]

                    w = gmap[:, :, :lf.shape[2], :lf.shape[3], :lf.shape[4]]
                    acc_fdi[slc] += lf * w
                    acc_jaw[slc] += lj * w
                    count[slc]   += w

                    rest_logits_list.append(lr)

    acc_fdi /= count.clamp(min=1e-6)
    acc_jaw /= count.clamp(min=1e-6)
    rest_logits = torch.stack(rest_logits_list).mean(0)

    return acc_fdi, acc_jaw, rest_logits


# ─── Post-processing ──────────────────────────────────────────────────────────
def largest_connected_component(mask: np.ndarray, label: int) -> np.ndarray:
    """Keep only the largest connected component for a given label."""
    try:
        from scipy.ndimage import label as scipy_label
    except ImportError:
        return mask  # graceful degradation

    binary = (mask == label).astype(np.uint8)
    labeled, num = scipy_label(binary)
    if num == 0:
        return mask
    sizes = [(labeled == i).sum() for i in range(1, num + 1)]
    largest = np.argmax(sizes) + 1
    mask[binary.astype(bool) & (labeled != largest)] = 0
    return mask


def postprocess_segmentation(seg_arr: np.ndarray,
                              apply_lcc: bool = True) -> np.ndarray:
    """Remove small islands per tooth class."""
    if not apply_lcc:
        return seg_arr
    for cls in range(1, 33):
        if (seg_arr == cls).sum() > 0:
            seg_arr = largest_connected_component(seg_arr, cls)
    return seg_arr


# ─── Label JSON builder ───────────────────────────────────────────────────────
def build_label_json(seg_arr: np.ndarray,
                     jaw_arr: np.ndarray,
                     rest_probs: np.ndarray) -> dict:
    """Build structured JSON with FDI labels, jaw assignment, and restoration flag."""
    teeth = []
    for cls in range(1, 33):
        voxels = int((seg_arr == cls).sum())
        if voxels == 0:
            continue
        fdi = idx_to_fdi(cls)

        # Jaw assignment: majority vote of jaw labels within tooth mask
        tooth_mask = seg_arr == cls
        jaw_vals = jaw_arr[tooth_mask]
        jaw_id   = int(np.bincount(jaw_vals.astype(int), minlength=3).argmax())

        teeth.append({
            "fdi_id":       fdi,
            "fdi_name":     FDI_NAMES.get(fdi, "Unknown"),
            "class_index":  cls,
            "jaw":          JAW_NAMES.get(jaw_id, "unknown"),
            "voxel_count":  voxels,
            "restoration_prob":  round(float(rest_probs[cls - 1]), 4),
            "restoration_flag":  bool(rest_probs[cls - 1] > 0.5),
        })

    return {"num_teeth": len(teeth), "teeth": teeth}


# ─── Main inference function ──────────────────────────────────────────────────
def run_inference(input_path: Path,
                  weights_path: Path,
                  output_dir: Path,
                  patch_size: tuple = (96, 96, 96),
                  overlap: float = 0.5,
                  apply_lcc: bool = True):

    try:
        from scripts.unet_preprocessing import preprocess
        from scripts.unet_training import CBCTSegNet
    except ImportError:
        from unet_preprocessing import preprocess
        from unet_training import CBCTSegNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Inference on %s | device=%s", input_path, device)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Preprocess ────────────────────────────────────────────────────────────
    pp_path = output_dir / "scan_preprocessed.nii.gz"
    orig_image = preprocess(input_path, pp_path)

    # ── Load model ────────────────────────────────────────────────────────────
    model = CBCTSegNet()
    if weights_path.exists():
        ckpt = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        log.info("Loaded weights from %s (epoch=%d, dice=%.4f)",
                 weights_path, ckpt.get("epoch", -1), ckpt.get("val_dice", 0))
    else:
        log.warning("Weights not found at %s – running with random init (demo only)",
                    weights_path)
    model.to(device)

    # ── Volume to tensor ──────────────────────────────────────────────────────
    import SimpleITK as sitk
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(pp_path))).astype(np.float32)
    volume = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

    # ── Sliding-window inference ──────────────────────────────────────────────
    logits_fdi, logits_jaw, logits_rest = sliding_window_inference(
        model, volume, patch_size, overlap, device)

    # ── Decode predictions ────────────────────────────────────────────────────
    seg_arr  = logits_fdi.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
    jaw_arr  = logits_jaw.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
    rest_prob = torch.sigmoid(logits_rest).squeeze().cpu().numpy()

    seg_arr = postprocess_segmentation(seg_arr, apply_lcc)

    # ── Save outputs ──────────────────────────────────────────────────────────
    ref_img = sitk.ReadImage(str(pp_path))

    seg_itk = sitk.GetImageFromArray(seg_arr)
    seg_itk.CopyInformation(ref_img)
    sitk.WriteImage(seg_itk, str(output_dir / "mask.nii.gz"), useCompression=True)
    log.info("Saved mask.nii.gz")

    jaw_itk = sitk.GetImageFromArray(jaw_arr)
    jaw_itk.CopyInformation(ref_img)
    sitk.WriteImage(jaw_itk, str(output_dir / "jaw_mask.nii.gz"), useCompression=True)
    log.info("Saved jaw_mask.nii.gz")

    labels = build_label_json(seg_arr, jaw_arr, rest_prob)
    label_path = output_dir / "labels.json"
    label_path.write_text(json.dumps(labels, indent=2))
    log.info("Saved labels.json  (%d teeth detected)", labels["num_teeth"])

    return labels


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",      type=Path, required=True)
    p.add_argument("--weights",    type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/demo"))
    p.add_argument("--patch-size", nargs=3, type=int, default=[96, 96, 96])
    p.add_argument("--overlap",    type=float, default=0.5)
    p.add_argument("--no-lcc",     action="store_true")
    args = p.parse_args()

    run_inference(
        input_path   = args.input,
        weights_path = args.weights,
        output_dir   = args.output_dir,
        patch_size   = tuple(args.patch_size),
        overlap      = args.overlap,
        apply_lcc    = not args.no_lcc,
    )
