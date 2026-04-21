import argparse
import json
import os
from collections import Counter

from tqdm import tqdm
import nibabel as nib
import numpy as np


def compute_inverse_frequency_weights(json_file, output_file, exclude_background=True):
    # Load class occurrences
    with open(json_file, "r") as f:
        class_occurrences = json.load(f)

    # Convert keys back to int (JSON keys are strings)
    class_occurrences = {int(k): v for k, v in class_occurrences.items()}

    # Optionally exclude background (class 0)
    if exclude_background and 0 in class_occurrences:
        del class_occurrences[0]

    if len(class_occurrences) == 0:
        raise ValueError("No classes found in distribution file")

    # Total number of samples is the max occurrence count
    total_samples = max(class_occurrences.values())

    # Compute weights as inverse frequency
    weight_array = np.zeros(max(class_occurrences.keys()) + 1, dtype=np.float32)
    for cls, count in class_occurrences.items():
        weight_array[cls] = total_samples / count if count > 0 else 0.0

    # Normalize so the mean of used classes is 1.0
    used = weight_array > 0
    if used.any():
        weight_array[used] *= (used.sum() / weight_array[used].sum())

    np.save(output_file, weight_array)
    print(f"Saved inverse frequency weights to {output_file}")
    return weight_array


def build_class_distribution(label_dir, output_json, mode="volume"):
    if not os.path.isdir(label_dir):
        raise ValueError(f"Label directory not found: {label_dir}")

    file_list = sorted(f for f in os.listdir(label_dir) if f.endswith(".nii.gz"))
    if not file_list:
        raise ValueError(f"No NIfTI files found in {label_dir}")

    if mode == "volume":
        counter = Counter()
        for fname in tqdm(file_list, desc="Counting voxels", unit="file"):
            path = os.path.join(label_dir, fname)
            img = nib.load(path)
            arr = np.asarray(img.dataobj, dtype=np.int64)
            counter.update(arr.ravel().tolist())
        class_distribution = {str(cls): int(count) for cls, count in sorted(counter.items())}
    elif mode == "presence":
        counter = Counter()
        for fname in tqdm(file_list, desc="Counting presence", unit="file"):
            path = os.path.join(label_dir, fname)
            img = nib.load(path)
            arr = np.asarray(img.dataobj, dtype=np.int64)
            classes_present = np.unique(arr)
            for cls in classes_present:
                counter[int(cls)] += 1
        class_distribution = {str(cls): int(count) for cls, count in sorted(counter.items())}
    else:
        raise ValueError("Unsupported mode. Choose 'volume' or 'presence'.")

    with open(output_json, "w") as f:
        json.dump(class_distribution, f, indent=2)

    print(f"Saved class distribution JSON ({mode}) to {output_json}")
    return class_distribution


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate class distribution and inverse-frequency class weights."
    )
    parser.add_argument(
        "--label-dir",
        default=None,
        help="Directory with label NIfTI files to compute class counts from."
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Path to save the computed class distribution JSON. If not set, selects a default based on mode."
    )
    parser.add_argument(
        "--mode",
        choices=["volume", "presence"],
        default="volume",
        help="Whether to count total voxel volume per class or number of files containing each class."
    )
    parser.add_argument(
        "--output-npy",
        default="data/class_invfreq_weights.npy",
        help="Path to save the generated inverse-frequency weights .npy file."
    )
    parser.add_argument(
        "--exclude-background",
        action="store_true",
        help="Exclude class 0 (background) when computing weights."
    )
    parser.add_argument(
        "--json-file",
        default=None,
        help="If provided, read an existing class distribution JSON instead of scanning labels."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_json is None:
        if args.mode == "volume":
            args.output_json = "data/class_distribution_allclasses.json"
        else:
            args.output_json = "data/class_distribution_allclasses_presence.json"

    if args.json_file is not None:
        if args.label_dir is not None:
            raise ValueError("Cannot specify both --label-dir and --json-file")
        json_file = args.json_file
    elif args.label_dir is not None:
        build_class_distribution(args.label_dir, args.output_json, mode=args.mode)
        json_file = args.output_json
    else:
        json_file = "data/class_distribution.json"

    compute_inverse_frequency_weights(
        json_file=json_file,
        output_file=args.output_npy,
        exclude_background=args.exclude_background
    )


if __name__ == "__main__":
    main()
