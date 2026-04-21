#!/usr/bin/env python3
"""
Generate distance + direction watershed maps for center-F labels
in the all-classes format (data/labelsAllClasses/).

Reads from:   data/labelsAllClasses/ToothFairy3F_*.nii.gz
Writes to:    data/deep_watershed_maps_allclasses/{direction_maps,distance_maps,distdir_maps}/

Run once before training:
    python scripts/generate_watershed_maps_centerF.py          # default: all F files, cuda
    python scripts/generate_watershed_maps_centerF.py --device cpu
    python scripts/generate_watershed_maps_centerF.py --center F P   # multiple centers
"""
import os
import sys
import re
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm

# allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_preparation import WatershedDistanceMapGenerator

LABEL_DIR = "data/labelsAllClasses"
OUTPUT_BASE = "data/deep_watershed_maps_allclasses"


def extract_center(fname: str) -> str:
    m = re.match(r"ToothFairy3([FSP])_", fname)
    if not m:
        raise ValueError(f"Cannot extract center from: {fname}")
    return m.group(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--center", nargs="*", default=["F"],
        help="Centers to process (default: F). Example: --center F P S"
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--label_dir", default=LABEL_DIR)
    parser.add_argument("--output_base", default=OUTPUT_BASE)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute even if output file already exists"
    )
    args = parser.parse_args()

    # Create output subdirs
    dir_map_dir = os.path.join(args.output_base, "direction_maps")
    dist_map_dir = os.path.join(args.output_base, "distance_maps")
    distdir_map_dir = os.path.join(args.output_base, "distdir_maps")
    for d in [dir_map_dir, dist_map_dir, distdir_map_dir]:
        os.makedirs(d, exist_ok=True)

    label_files = sorted(f for f in os.listdir(args.label_dir) if f.endswith(".nii.gz"))
    label_files = [f for f in label_files if extract_center(f) in args.center]
    print(f"Processing {len(label_files)} files from centers {args.center} using device={args.device}")

    gen = WatershedDistanceMapGenerator(device=args.device)

    for fname in tqdm(label_files, desc="Generating watershed maps"):
        out_distdir = os.path.join(distdir_map_dir, fname)
        if not args.overwrite and os.path.exists(out_distdir):
            continue

        label_path = os.path.join(args.label_dir, fname)
        nib_vol = nib.load(label_path)
        label_volume = np.asanyarray(nib_vol.dataobj, dtype=np.int32)

        # Distance map (EDT per class, normalised)
        dist_map = gen.get_edt_map(label_volume)

        # Direction map (3-channel, channel-first)
        dir_map_cf = gen.get_dir_map(dist_map)           # shape (3, H, W, D)
        dir_map_cl = np.moveaxis(dir_map_cf, 0, -1)      # shape (H, W, D, 3)

        # ---- Save distance map (3D float32) ----
        nib.save(
            nib.Nifti1Image(dist_map.astype(np.float32), nib_vol.affine),
            os.path.join(dist_map_dir, fname)
        )

        # ---- Save direction map (4D: H,W,D,3) ----
        hdr_dir = _make_4d_header(nib_vol, n_channels=3, dtype=np.float32)
        hdr_dir["cal_min"] = -1.0
        hdr_dir["cal_max"] = 1.0
        nib.save(
            nib.Nifti1Image(dir_map_cl.astype(np.float32), nib_vol.affine, hdr_dir),
            os.path.join(dir_map_dir, fname)
        )

        # ---- Save combined distdir map (4D: H,W,D,4 = dist+dir) ----
        dist_ch = dist_map[..., np.newaxis]                      # (H,W,D,1)
        combined = np.concatenate([dist_ch, dir_map_cl], axis=-1)  # (H,W,D,4)
        hdr_distdir = _make_4d_header(nib_vol, n_channels=4, dtype=np.float32)
        nib.save(
            nib.Nifti1Image(combined.astype(np.float32), nib_vol.affine, hdr_distdir),
            out_distdir
        )

    print(f"Done. Maps written to: {args.output_base}")


def _make_4d_header(src_nib, n_channels: int, dtype):
    """Create a NIfTI header for a 4-D volume derived from src_nib."""
    hdr = nib.Nifti1Header()
    src_hdr = src_nib.header
    for key in [
        "pixdim", "xyzt_units", "qform_code", "sform_code",
        "quatern_b", "quatern_c", "quatern_d",
        "qoffset_x", "qoffset_y", "qoffset_z",
        "srow_x", "srow_y", "srow_z",
    ]:
        hdr[key] = src_hdr[key]

    shape = src_nib.shape[:3]
    hdr.set_data_dtype(dtype)
    hdr["dim"][0] = 4
    hdr["dim"][1] = shape[0]
    hdr["dim"][2] = shape[1]
    hdr["dim"][3] = shape[2]
    hdr["dim"][4] = n_channels
    hdr["intent_code"] = 2001       # NIFTI_INTENT_TIME_SERIES
    hdr["intent_name"] = b"Vector"
    hdr["scl_slope"] = 1.0
    hdr["scl_inter"] = 0.0
    return hdr


if __name__ == "__main__":
    main()
