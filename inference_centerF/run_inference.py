"""
run_inference.py — centerF CBCT segmentation inference

Usage:
    # Single file:
    python run_inference.py --input scan.nii.gz --output ./predictions

    # Directory (all .nii.gz files):
    python run_inference.py --input ./cbct_scans/ --output ./predictions

    # Override checkpoint path:
    python run_inference.py --input scan.nii.gz --output ./predictions \\
        --checkpoint checkpoints/my_model.pth

    # Keep internal class indices (0-46) instead of FDI/challenge labels:
    python run_inference.py --input scan.nii.gz --output ./predictions --no-remap


Output:
    One .nii.gz segmentation file per input scan, saved to --output.
    Filename: <input_stem>_seg.nii.gz

Dependencies (conda/pip):
    torch, monai, nibabel, SimpleITK (optional)
"""
import argparse
import os
import sys
import time
from pathlib import Path

import dataclasses
import numpy as np
import nibabel as nib
import torch

# ── Local imports (run from the inference_centerF/ folder) ────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config import Args
from src.transforms import Transforms
from src.inference import run_inference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collect_inputs(input_path: str) -> list:
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted(p.glob("*.nii.gz")) + sorted(p.glob("*.nii"))
        if not files:
            raise FileNotFoundError(f"No .nii.gz / .nii files found in {p}")
        return files
    raise FileNotFoundError(f"Input path does not exist: {p}")


def save_nifti(array: np.ndarray, ref_nib: nib.Nifti1Image, output_path: Path):
    """Save array with the same header/affine as the reference image."""
    out_img = nib.Nifti1Image(array.astype(np.int32), ref_nib.affine, ref_nib.header)
    out_img.set_data_dtype(np.int32)
    nib.save(out_img, str(output_path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="centerF CBCT segmentation inference")
    parser.add_argument("--input",      required=True, help="Input .nii.gz file or directory")
    parser.add_argument("--output",     required=True, help="Output directory for segmentations")
    parser.add_argument("--checkpoint", default=None,  help="Path to checkpoint .pth (overrides config)")
    parser.add_argument("--no-remap",   action="store_true",
                        help="Keep internal class indices 0-46 (skip FDI/challenge remapping)")
    parser.add_argument("--pixdim",     default=None,  type=float,
                        help="Override voxel spacing in mm (default: 0.6 from config)")
    opt = parser.parse_args()

    # ── Build args ─────────────────────────────────────────────────────────
    args = Args()
    if opt.checkpoint:
        args = dataclasses.replace(args, checkpoint_path=opt.checkpoint)
    if opt.no_remap:
        args = dataclasses.replace(args, remap_to_challenge_labels=False)
    if opt.pixdim is not None:
        args = dataclasses.replace(args, pixdim=opt.pixdim)

    device = get_device()
    print(f"Device       : {device}")
    print(f"Checkpoint   : {args.checkpoint_path}")
    print(f"pixdim       : {args.pixdim} mm")
    print(f"Remap labels : {args.remap_to_challenge_labels}")

    if not Path(args.checkpoint_path).exists():
        print(f"\nERROR: checkpoint not found: {args.checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(opt.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = collect_inputs(opt.input)
    print(f"\nFound {len(input_files)} file(s) to process.\n")

    transforms = Transforms(args, device=device)

    total_t0 = time.time()
    errors = []

    for idx, input_file in enumerate(input_files, 1):
        stem = input_file.name.replace(".nii.gz", "").replace(".nii", "")
        output_file = output_dir / f"{stem}_seg.nii.gz"
        print(f"[{idx}/{len(input_files)}] {input_file.name}")

        try:
            # ── Preprocess ────────────────────────────────────────────────
            data = transforms.preprocess({"image": str(input_file)})
            print(f"  Preprocessed shape : {data['image'].shape}")

            # Add batch dimension: (1,H,W,D) → (1,1,H,W,D)
            data["image"] = data["image"].unsqueeze(0)

            # Load original for header preservation
            ref_nib = nib.load(str(input_file))

            # ── Inference ─────────────────────────────────────────────────
            seg = run_inference(data["image"], args, device, transforms)
            # seg: (H_orig, W_orig, D_orig) np.int32 in original geometry

            print(f"  Output shape       : {seg.shape}")
            print(f"  Label range        : [{seg.min()}, {seg.max()}]")

            # ── Save ──────────────────────────────────────────────────────
            # Transpose to match NIfTI convention (D, H, W) for nibabel
            seg_save = np.transpose(seg, (2, 1, 0))
            out_img = nib.Nifti1Image(seg_save.astype(np.int32), ref_nib.affine, ref_nib.header)
            out_img.set_data_dtype(np.int32)
            nib.save(out_img, str(output_file))
            print(f"  Saved → {output_file}")

        except Exception as exc:
            print(f"  ERROR: {exc}")
            errors.append((input_file.name, exc))

        # Free GPU memory between cases
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print()

    elapsed = time.time() - total_t0
    print("=" * 60)
    print(f"Done. {len(input_files) - len(errors)}/{len(input_files)} succeeded in {elapsed:.0f}s.")
    if errors:
        print(f"\nFailed ({len(errors)}):")
        for name, exc in errors:
            print(f"  {name}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
