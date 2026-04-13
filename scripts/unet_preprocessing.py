

import argparse
import os
import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
TARGET_SPACING_MM = (0.4, 0.4, 0.4)  # isotropic 0.4 mm
HU_CLIP = (-1000, 3000)              # Hounsfield range relevant for CBCT / bone
NORMALISE_MEAN = 0.0
NORMALISE_STD  = 1.0


# ─── I/O helpers ──────────────────────────────────────────────────────────────
def load_volume(path: Path) -> sitk.Image:
    """Load any supported CBCT format into a SimpleITK Image."""
    path = Path(path)
    if path.is_dir():
        log.info("Reading DICOM series from %s", path)
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(path))
        if not dicom_names:
            raise ValueError(f"No DICOM series found in {path}")
        reader.SetFileNames(dicom_names)
        return reader.Execute()
    else:
        suffix = "".join(path.suffixes).lower()
        if suffix not in {".mha", ".nii", ".nii.gz", ".gz"}:
            log.warning("Unexpected extension %s – attempting to load anyway", suffix)
        log.info("Reading volume from %s", path)
        return sitk.ReadImage(str(path), sitk.sitkFloat32)


def save_volume(image: sitk.Image, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(out_path), useCompression=True)
    log.info("Saved → %s", out_path)


# ─── Preprocessing steps ──────────────────────────────────────────────────────
def resample_isotropic(image: sitk.Image,
                        target_spacing: tuple = TARGET_SPACING_MM,
                        interpolator=sitk.sitkBSpline) -> sitk.Image:
    """Resample to isotropic voxel spacing using B-spline interpolation."""
    original_spacing = image.GetSpacing()
    original_size    = image.GetSize()

    new_size = [
        int(round(orig_sz * orig_sp / tgt_sp))
        for orig_sz, orig_sp, tgt_sp
        in zip(original_size, original_spacing, target_spacing)
    ]

    log.info(
        "Resampling %s @ %.2fmm³ → %s @ %.2fmm³",
        original_size, original_spacing[0],
        new_size,       target_spacing[0],
    )

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(-1000)
    resampler.SetInterpolator(interpolator)
    return resampler.Execute(image)


def clip_and_normalise(image: sitk.Image,
                        hu_min: float = HU_CLIP[0],
                        hu_max: float = HU_CLIP[1]) -> sitk.Image:
    """Clip HU values then z-score normalise."""
    arr = sitk.GetArrayFromImage(image).astype(np.float32)

    arr = np.clip(arr, hu_min, hu_max)

    mean = arr.mean()
    std  = arr.std() + 1e-8
    arr  = (arr - mean) / std

    log.info("HU clipped to [%d, %d], normalised μ=%.3f σ=%.3f",
             hu_min, hu_max, mean, std)

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image)
    return out


def pad_to_divisible(image: sitk.Image, divisor: int = 16) -> sitk.Image:
    """Zero-pad all axes so each dimension is divisible by `divisor`."""
    size = list(image.GetSize())        # (W, H, D) in SimpleITK order
    new_size = [int(np.ceil(s / divisor) * divisor) for s in size]
    pad_lower = [0, 0, 0]
    pad_upper = [n - s for n, s in zip(new_size, size)]

    if pad_upper == [0, 0, 0]:
        return image

    log.info("Padding %s → %s (divisor=%d)", size, new_size, divisor)
    padded = sitk.ConstantPad(image, pad_lower, pad_upper, 0.0)
    return padded


# ─── Full pipeline ─────────────────────────────────────────────────────────────
def preprocess(
    input_path:  Path,
    output_path: Path,
    target_spacing: tuple = TARGET_SPACING_MM,
    hu_clip:    tuple = HU_CLIP,
    pad_divisor: int  = 16,
) -> sitk.Image:
    """Run the complete preprocessing pipeline on one volume."""
    image = load_volume(input_path)

    # Cast to float32 early for consistent arithmetic
    image = sitk.Cast(image, sitk.sitkFloat32)

    image = resample_isotropic(image, target_spacing)
    image = clip_and_normalise(image, *hu_clip)
    image = pad_to_divisible(image, pad_divisor)

    save_volume(image, output_path)
    return image


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Preprocess CBCT volumes for segmentation.")
    p.add_argument("input",  type=Path, help="Input volume (.mha/.nii/.nii.gz) or DICOM dir")
    p.add_argument("output", type=Path, help="Output .nii.gz path")
    p.add_argument("--spacing", nargs=3, type=float, default=list(TARGET_SPACING_MM),
                   metavar=("X", "Y", "Z"), help="Target voxel spacing in mm")
    p.add_argument("--hu-min",  type=float, default=HU_CLIP[0])
    p.add_argument("--hu-max",  type=float, default=HU_CLIP[1])
    p.add_argument("--pad-divisor", type=int, default=16)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(
        input_path   = args.input,
        output_path  = args.output,
        target_spacing = tuple(args.spacing),
        hu_clip      = (args.hu_min, args.hu_max),
        pad_divisor  = args.pad_divisor,
    )
