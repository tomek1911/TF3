#!/usr/bin/env python3
"""
Label preprocessing for the "all classes in single model" setup.

Produces single-channel 3D NIfTI files where:
  - Background:    0
  - Anatomy:       1-10  (unchanged from original)
  - Teeth:         11-48 (original FDI IDs preserved; pulp voxels are NOT folded in)
  - Canals:        103-105
  - Merged pulp:   100   (all pulp classes 111-148 are collapsed into class 100)

The value 100 is used as a sentinel so it doesn't collide with any real tooth ID (11-48)
or canal ID (103-105). The Transforms remapping in transforms.py then maps 100 -> 46
to get a contiguous 47-class label (0-46).

Usage:
    python src/label_preprocessing_all_classes.py          # process all centers
    python src/label_preprocessing_all_classes.py --center F   # only center F
"""
import os
import re
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm

INPUT_DIR = "data/labelsTr"
OUTPUT_DIR = "data/labelsAllClasses"

PULP_IDS = set(range(111, 149))   # 111-148  (pulp for each of the 32 FDI tooth IDs)
PULP_SENTINEL = 100               # value written to merged-pulp voxels in output


def remap_to_allclasses(data: np.ndarray) -> np.ndarray:
    """
    Convert a raw label array to the all-classes format.

    Pulp voxels (111-148) are collapsed to PULP_SENTINEL (100).
    All other valid label values are kept unchanged.
    Anything not in the valid set is mapped to background (0).

    Valid non-pulp values: 0, 1-10, 11-48, 103-105.
    """
    data = data.astype(np.int32, copy=False)

    valid_non_pulp = (
        set(range(0, 11))       # background + anatomy
        | set(range(11, 49))    # teeth (FDI)
        | {103, 104, 105}       # inferior alveolar / maxillary / incisive canals
    )

    pulp_mask = np.isin(data, list(PULP_IDS))
    invalid_mask = ~(np.isin(data, list(valid_non_pulp)) | pulp_mask)

    output = data.copy()
    output[pulp_mask] = PULP_SENTINEL
    output[invalid_mask] = 0          # zero out anything unknown
    return output.astype(np.int32)


def extract_center(fname: str) -> str:
    m = re.match(r"ToothFairy3([FSP])_", fname)
    if not m:
        raise ValueError(f"Cannot extract center from: {fname}")
    return m.group(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--center", nargs="*", default=None,
        help="Restrict processing to these center letters, e.g. --center F  or --center F P"
    )
    parser.add_argument("--input_dir", default=INPUT_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    label_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".nii.gz"))

    if args.center:
        label_files = [f for f in label_files if extract_center(f) in args.center]
        print(f"Filtered to centers {args.center}: {len(label_files)} files.")

    for fname in tqdm(label_files, desc="Preprocessing labels (all classes)"):
        path_in = os.path.join(args.input_dir, fname)
        path_out = os.path.join(args.output_dir, fname)

        if os.path.exists(path_out):
            continue  # skip already processed files

        nib_img = nib.load(path_in)
        data = np.asarray(nib_img.dataobj, dtype=np.int32)

        remapped = remap_to_allclasses(data)

        out_img = nib.Nifti1Image(remapped.astype(np.int16), nib_img.affine, nib_img.header)
        nib.save(out_img, path_out)

    print(f"Done. Labels saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
