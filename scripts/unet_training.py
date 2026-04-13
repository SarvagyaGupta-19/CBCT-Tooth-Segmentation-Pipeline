

import argparse
import json
import logging
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ─── Hyper-parameters ─────────────────────────────────────────────────────────
NUM_FDI_CLASSES  = 33   # background + 32 FDI tooth IDs (11-48)
NUM_JAW_CLASSES  = 3    # background, mandible, maxilla
PATCH_SIZE       = (96, 96, 96)
BATCH_SIZE       = 2
LEARNING_RATE    = 3e-4
WEIGHT_DECAY     = 1e-5
NUM_WORKERS      = 4
VAL_INTERVAL     = 10   # validate every N epochs
SEED             = 42


# ─── Reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─── Dataset ──────────────────────────────────────────────────────────────────
class ToothFairyDataset(Dataset):
    """
    Expects a folder structure:
        data_dir/
          imagesTr/<case_id>.nii.gz
          labelsTr/<case_id>.nii.gz          # FDI integer labels 0-32
          jaw_masksTr/<case_id>.nii.gz       # 0=bg, 1=mandible, 2=maxilla
          restoration_labels.json            # {case_id: {tooth_id: 0|1}}
          dataset.json                        # official ToothFairy2 metadata
    """

    def __init__(self, data_dir: Path, split: str = "train",
                 patch_size: tuple = PATCH_SIZE, augment: bool = True):
        self.data_dir   = Path(data_dir)
        self.patch_size = patch_size
        self.augment    = augment and (split == "train")

        split_file = self.data_dir / f"splits_{split}.json"
        if split_file.exists():
            with open(split_file) as f:
                self.cases = json.load(f)
        else:
            all_cases = sorted([p.stem.replace(".nii", "") for p
                                 in (self.data_dir / "imagesTr").glob("*.nii.gz")])
            rng = np.random.default_rng(SEED)
            rng.shuffle(all_cases)
            n = len(all_cases)
            train_ids = all_cases[:int(0.7 * n)]
            val_ids   = all_cases[int(0.7 * n):int(0.85 * n)]
            test_ids  = all_cases[int(0.85 * n):]
            splits = {"train": train_ids, "val": val_ids, "test": test_ids}
            for sname, slist in splits.items():
                with open(self.data_dir / f"splits_{sname}.json", "w") as f:
                    json.dump(slist, f, indent=2)
                log.info("Created split '%s': %d cases", sname, len(slist))
            self.cases = splits[split]

        rest_path = self.data_dir / "restoration_labels.json"
        self.restoration = json.loads(rest_path.read_text()) if rest_path.exists() else {}

        log.info("Dataset [%s]: %d cases", split, len(self.cases))

    def __len__(self): return len(self.cases)

    def __getitem__(self, idx):
        import SimpleITK as sitk  # lazy import to keep startup fast
        case_id = self.cases[idx]

        img  = self._load(self.data_dir / "imagesTr"   / f"{case_id}.nii.gz")
        seg  = self._load(self.data_dir / "labelsTr"   / f"{case_id}.nii.gz",
                          dtype=np.int64)
        jaw  = self._load(self.data_dir / "jaw_masksTr" / f"{case_id}.nii.gz",
                          dtype=np.int64)

        # Random patch crop
        img, seg, jaw = self._random_crop(img, seg, jaw)

        # Augmentation
        if self.augment:
            img, seg, jaw = self._augment(img, seg, jaw)

        # Build per-tooth restoration vector (32 dims, binary)
        rest_vec = np.zeros(32, dtype=np.float32)
        if case_id in self.restoration:
            for fdi_str, label in self.restoration[case_id].items():
                fdi  = int(fdi_str)
                tidx = self._fdi_to_idx(fdi)
                if 0 <= tidx < 32:
                    rest_vec[tidx] = float(label)

        return {
            "image":       torch.from_numpy(img[None]).float(),   # (1, D, H, W)
            "seg":         torch.from_numpy(seg).long(),           # (D, H, W)
            "jaw":         torch.from_numpy(jaw).long(),           # (D, H, W)
            "restoration": torch.from_numpy(rest_vec),            # (32,)
            "case_id":     case_id,
        }

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _load(path: Path, dtype=np.float32):
        import SimpleITK as sitk
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        return arr.astype(dtype)

    def _random_crop(self, img, seg, jaw):
        D, H, W = img.shape
        pd, ph, pw = self.patch_size
        d0 = random.randint(0, max(0, D - pd))
        h0 = random.randint(0, max(0, H - ph))
        w0 = random.randint(0, max(0, W - pw))
        slc = np.s_[d0:d0+pd, h0:h0+ph, w0:w0+pw]
        # Pad if volume smaller than patch
        def _pad(a, target, val=0):
            pad = [(0, max(0, t - s)) for s, t in zip(a.shape, target)]
            return np.pad(a, pad, constant_values=val)
        img = _pad(img[slc], self.patch_size, val=0.0)
        seg = _pad(seg[slc], self.patch_size, val=0)
        jaw = _pad(jaw[slc], self.patch_size, val=0)
        return img, seg, jaw

    @staticmethod
    def _augment(img, seg, jaw):
        """Simple intensity + flip augmentation (no external deps)."""
        # Random flips
        for axis in range(3):
            if random.random() > 0.5:
                img = np.flip(img, axis=axis).copy()
                seg = np.flip(seg, axis=axis).copy()
                jaw = np.flip(jaw, axis=axis).copy()
        # Intensity jitter
        img = img + np.random.normal(0, 0.05)
        img = img * np.random.uniform(0.9, 1.1)
        return img, seg, jaw

    @staticmethod
    def _fdi_to_idx(fdi: int) -> int:
        """Map FDI tooth number (11-48) to 0-based index."""
        quadrant = (fdi // 10) - 1
        position = (fdi %  10) - 1
        return quadrant * 8 + position


# ─── Loss ─────────────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        one_hot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1)
        dims = (0, 2, 3, 4)
        intersection = (probs * one_hot).sum(dims)
        cardinality   = (probs + one_hot).sum(dims)
        dice = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    """Combined Dice + Cross-Entropy for segmentation heads."""
    def __init__(self, num_classes: int, ce_weight: float = 0.5):
        super().__init__()
        self.dice = DiceLoss(num_classes)
        self.ce   = nn.CrossEntropyLoss()
        self.w    = ce_weight

    def forward(self, logits, targets):
        return self.dice(logits, targets) + self.w * self.ce(logits, targets)


# ─── Minimal U-Net baseline (used when nnU-Net env not configured) ─────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )
    def forward(self, x): return self.block(x)


class CBCTSegNet(nn.Module):
    """
    Lightweight 3-D U-Net with three output heads:
      1. FDI segmentation   (33 classes)
      2. Jaw segmentation   (3 classes)
      3. Restoration score  (32 sigmoid logits, one per tooth)
    For production, replace with nnU-Net v2 via `nnUNetv2_train`.
    """
    FEATURES = [32, 64, 128, 256]

    def __init__(self):
        super().__init__()
        F = self.FEATURES

        # Encoder
        self.enc = nn.ModuleList([ConvBlock(1 if i == 0 else F[i-1], F[i])
                                   for i in range(len(F))])
        self.pool = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(F[-1], F[-1] * 2)

        # Decoder
        self.up   = nn.ModuleList([nn.ConvTranspose3d(F[-1]*2 if i==0 else F[-(i)]*2,
                                                       F[-(i+1)], 2, 2)
                                    for i in range(len(F))])
        self.dec  = nn.ModuleList([ConvBlock(F[-(i+1)]*2, F[-(i+1)])
                                    for i in range(len(F))])

        # Output heads
        self.head_fdi  = nn.Conv3d(F[0], NUM_FDI_CLASSES, 1)
        self.head_jaw  = nn.Conv3d(F[0], NUM_JAW_CLASSES,  1)
        self.gap       = nn.AdaptiveAvgPool3d(1)
        self.head_rest = nn.Linear(F[0], 32)

    def forward(self, x):
        skips = []
        for enc_layer in self.enc:
            x = enc_layer(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for i, (up, dec) in enumerate(zip(self.up, self.dec)):
            x = up(x)
            skip = skips[-(i+1)]
            # Handle size mismatch from uneven pooling
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="trilinear",
                                               align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))
        logits_fdi  = self.head_fdi(x)
        logits_jaw  = self.head_jaw(x)
        feat_pool   = self.gap(x).flatten(1)
        logits_rest = self.head_rest(feat_pool)
        return logits_fdi, logits_jaw, logits_rest


# ─── Training loop ────────────────────────────────────────────────────────────
def train(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Datasets
    train_ds = ToothFairyDataset(args.data_dir, "train", augment=True)
    val_ds   = ToothFairyDataset(args.data_dir, "val",   augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    model = CBCTSegNet().to(device)
    log.info("Model parameters: %s M", f"{sum(p.numel() for p in model.parameters())/1e6:.1f}")

    # Optimiser & scheduler
    opt  = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    # Loss functions
    seg_loss_fdi = SegmentationLoss(NUM_FDI_CLASSES)
    seg_loss_jaw = SegmentationLoss(NUM_JAW_CLASSES)
    rest_loss    = nn.BCEWithLogitsLoss()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        for batch in train_dl:
            img  = batch["image"].to(device)
            seg  = batch["seg"].to(device)
            jaw  = batch["jaw"].to(device)
            rest = batch["restoration"].to(device)

            opt.zero_grad()
            logits_fdi, logits_jaw, logits_rest = model(img)

            loss = (seg_loss_fdi(logits_fdi, seg)
                    + 0.5 * seg_loss_jaw(logits_jaw, jaw)
                    + 0.3 * rest_loss(logits_rest, rest))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(loss.item())

        sched.step()
        avg_loss = np.mean(train_losses)

        # ── Validate ─────────────────────────────────────────────────────────
        if epoch % VAL_INTERVAL == 0:
            model.eval()
            dice_scores = []
            with torch.no_grad():
                for batch in val_dl:
                    img = batch["image"].to(device)
                    seg = batch["seg"].to(device)
                    logits_fdi, _, _ = model(img)
                    pred = logits_fdi.argmax(dim=1)
                    # Mean Dice over FDI classes (excluding background)
                    for c in range(1, NUM_FDI_CLASSES):
                        pred_c = (pred == c).float()
                        gt_c   = (seg  == c).float()
                        inter  = (pred_c * gt_c).sum()
                        union  = pred_c.sum() + gt_c.sum()
                        if union > 0:
                            dice_scores.append((2 * inter / union).item())

            val_dice = np.mean(dice_scores) if dice_scores else 0.0
            log.info("Epoch %d/%d | loss=%.4f | val_dice=%.4f",
                     epoch, args.epochs, avg_loss, val_dice)

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                ckpt_path = out_dir / "best_model.pth"
                torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                             "val_dice": val_dice, "optimizer": opt.state_dict()},
                            ckpt_path)
                log.info("  ✓ New best checkpoint saved (dice=%.4f)", val_dice)
        else:
            log.info("Epoch %d/%d | loss=%.4f", epoch, args.epochs, avg_loss)

    log.info("Training complete. Best val dice: %.4f", best_val_dice)


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("runs/cbct_seg"))
    p.add_argument("--epochs",     type=int,  default=300)
    p.add_argument("--fold",       type=int,  default=0,
                   help="Cross-validation fold (0-4). Used for nnU-Net mode.")
    train(p.parse_args())
