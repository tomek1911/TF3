import argparse
import json
import os
from collections import Counter

from tqdm import tqdm
import nibabel as nib
import numpy as np


def compute_inverse_frequency_weights(json_file, output_file, output_json=None, exclude_background=True):
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

    np.save(output_file, weight_array)
    print(f"Saved inverse frequency weights to {output_file}")

    if output_json is None:
        output_json = os.path.splitext(output_file)[0] + ".json"

    weights_dict = {str(idx): float(weight) for idx, weight in enumerate(weight_array)}
    with open(output_json, "w") as f:
        json.dump(weights_dict, f, indent=2)
    print(f"Saved inverse frequency weights JSON to {output_json}")

    return weight_array


def _raw_inverse_freq(class_occurrences: dict) -> np.ndarray:
    """Compute unnormalized inverse-frequency weights from a {class_idx: count} dict."""
    total = max(class_occurrences.values())
    n_classes = max(class_occurrences.keys()) + 1
    w = np.zeros(n_classes, dtype=np.float64)
    for cls, count in class_occurrences.items():
        w[cls] = total / count if count > 0 else 0.0
    return w


def _median_freq(class_occurrences: dict) -> np.ndarray:
    """
    Median-frequency balancing (Eigen & Fergus 2015).
    w_c = median(freq) / freq_c, where freq_c = count_c / total_count.
    Anchors the scale so the median class gets weight ~1.
    Produces far smaller values than raw inverse-frequency for highly
    imbalanced classes (e.g. background vs. canals in CBCT).
    """
    total = sum(class_occurrences.values())
    n_classes = max(class_occurrences.keys()) + 1
    freq = np.zeros(n_classes, dtype=np.float64)
    for cls, count in class_occurrences.items():
        freq[cls] = count / total
    used_freq = freq[freq > 0]
    med = np.median(used_freq)
    w = np.zeros(n_classes, dtype=np.float64)
    for cls in class_occurrences:
        w[cls] = med / freq[cls] if freq[cls] > 0 else 0.0
    return w


def compute_combined_weights(
    presence_json: str,
    volume_json: str,
    output_npy: str,
    output_json: str | None = None,
    combination: str = "geometric_mean",   # "geometric_mean" | "presence_only" | "volume_only"
    weighting_fn: str = "inverse_freq",    # "inverse_freq" | "median_freq"
    log_damping: bool = False,             # apply log(1 + w) before combining / post-processing
    w_max: float | None = None,            # hard upper cap (None = no cap)
    w_min: float | None = None,            # hard lower cap for classes with w > 0 (None = no floor)
    exclude_background: bool = True,
) -> np.ndarray:
    """
    Build CE class weights from presence and/or volume inverse-frequency counts.

    combination options
    -------------------
    geometric_mean  : sqrt(w_presence * w_volume)  — balances both signals
    presence_only   : w_presence only
    volume_only     : w_volume only

    weighting_fn options
    --------------------
    inverse_freq    : w_c = max_count / count_c  — aggressive, raw ratio
    median_freq     : w_c = median(freq) / freq_c — anchored at median class,
                      stays in reasonable range without needing a cap

    Post-processing order (all optional)
    -------------------------------------
    1. log_damping  : w = log(1 + w)   — compresses extreme ratios
    2. w_max cap    : w = min(w, w_max)
    3. w_min floor  : w[w > 0] = max(w[w > 0], w_min)  — background (0) stays 0

    No renormalization is applied — the raw ratio is preserved so the
    optimizer sees the true relative class difficulty.
    """

    def _load(path):
        with open(path) as f:
            d = json.load(f)
        return {int(k): v for k, v in d.items()}

    pres = _load(presence_json)
    vol  = _load(volume_json)

    if exclude_background:
        pres.pop(0, None)
        vol.pop(0, None)

    n_classes = max(max(pres.keys(), default=0), max(vol.keys(), default=0)) + 1

    _wfn = _median_freq if weighting_fn == "median_freq" else _raw_inverse_freq
    w_pres = _wfn(pres) if pres else np.zeros(n_classes, np.float64)
    w_vol  = _wfn(vol)  if vol  else np.zeros(n_classes, np.float64)

    # --- combine ---
    if combination == "geometric_mean":
        w = np.sqrt(w_pres * w_vol)
    elif combination == "presence_only":
        w = w_pres.copy()
    elif combination == "volume_only":
        w = w_vol.copy()
    else:
        raise ValueError(f"Unknown combination mode: {combination!r}")

    # --- log damping ---
    if log_damping:
        w = np.log1p(w)

    # --- hard caps ---
    if w_max is not None:
        w = np.minimum(w, w_max)
    if w_min is not None:
        used = w > 0
        w[used] = np.maximum(w[used], w_min)

    w = w.astype(np.float32)

    np.save(output_npy, w)
    print(f"Saved combined weights ({combination}, {weighting_fn}"
          f"{', log-damped' if log_damping else ''}"
          f"{f', cap={w_max}' if w_max else ''}"
          f"{f', floor={w_min}' if w_min else ''}) to {output_npy}")

    if output_json is None:
        output_json = os.path.splitext(output_npy)[0] + ".json"
    with open(output_json, "w") as f:
        json.dump({str(i): float(v) for i, v in enumerate(w)}, f, indent=2)
    print(f"Saved combined weights JSON to {output_json}")

    return w


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
        "--output-json-weights",
        default=None,
        help="Path to save the generated inverse-frequency weights as JSON. Defaults to the .npy path with .json extension."
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
    # --- combined-weights options ---
    parser.add_argument(
        "--combine",
        default=None,
        choices=["geometric_mean", "presence_only", "volume_only"],
        help="If set, compute combined weights from both presence and volume JSONs instead of single-mode weights."
    )
    parser.add_argument(
        "--presence-json",
        default=None,
        help="Presence distribution JSON (required when --combine is set)."
    )
    parser.add_argument(
        "--volume-json",
        default=None,
        help="Volume distribution JSON (required when --combine is set)."
    )
    parser.add_argument(
        "--log-damping",
        action="store_true",
        help="Apply log(1 + w) to weights before capping/normalizing."
    )
    parser.add_argument(
        "--w-max",
        type=float,
        default=None,
        help="Hard upper cap on weight values (applied after optional log-damping)."
    )
    parser.add_argument(
        "--w-min",
        type=float,
        default=None,
        help="Hard lower floor for classes with weight > 0 (keeps background at 0)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- combined weights path ---
    if args.combine is not None:
        if not args.presence_json or not args.volume_json:
            raise ValueError("--presence-json and --volume-json are required when --combine is set.")
        compute_combined_weights(
            presence_json=args.presence_json,
            volume_json=args.volume_json,
            output_npy=args.output_npy,
            output_json=args.output_json_weights,
            combination=args.combine,
            log_damping=args.log_damping,
            w_max=args.w_max,
            w_min=args.w_min,
            exclude_background=args.exclude_background,
        )
        return

    # --- single-mode (presence or volume) path ---
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
        output_json=args.output_json_weights,
        exclude_background=args.exclude_background
    )


if __name__ == "__main__":
    main()
