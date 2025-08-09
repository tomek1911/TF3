import os
import json
import re
from collections import defaultdict, Counter
from torch.utils.data import SequentialSampler, RandomSampler, WeightedRandomSampler

def load_dataset_json(json_path):
    with open(json_path, 'r') as f:
        dataset_info = json.load(f)
    return dataset_info.get('labels', {})

def extract_device(filename):
    """
    Extract device source letter (F, S, or P) from filename pattern 'ToothFairy3X_...'
    """
    match = re.match(r'ToothFairy3([FSP])_', filename)
    if not match:
        raise ValueError(f"Filename {filename} does not match expected pattern")
    return match.group(1)

def split_train_val(args):
    """
    Split filenames in args.data directory into train and val sets.
    Select `args.val_items` files per device for validation.
    Returns two lists of filenames.
    """
    images_dir = os.path.join(args.data, "imagesTr")
    labels_dir = os.path.join(args.data, "labelsTr")

    device_files = defaultdict(list)
    # List image files only
    for fname in sorted(os.listdir(images_dir)):
        try:
            device = extract_device(fname)
            device_files[device].append(fname)
        except ValueError:
            continue

    val_files = []
    train_files = []
    for device, files in device_files.items():
        val_files.extend(files[:args.val_items])
        train_files.extend(files[args.val_items:])
        
    # Pair image and label files - remove '_0000.nii.gz' from image filenames to match label filenames
    def pair_files(file_list):
        return [
            {"image": os.path.join(images_dir, f), "label": os.path.join(labels_dir, f.replace("_0000.nii.gz", ".nii.gz"))}
            for f in file_list
        ]

    return pair_files(train_files), pair_files(val_files)

def create_domain_labels(file_list):
    """
    Given a list of filenames, return list of corresponding device labels.
    """
    return [extract_device(os.path.basename(item['image'])) for item in file_list]

def build_sampler(dataset, domain_labels, args):
    """
    Create a sampler based on args.sampler choice.

    args.sampler can be:
        'sequential'     - SequentialSampler
        'random'   - RandomSampler
        'weighted' - WeightedRandomSampler

    """
    sampler_type = args.sampler_type.lower()
    if sampler_type == 'sequential':
        return SequentialSampler(dataset)
    elif sampler_type == 'random':
        return RandomSampler(dataset)
    elif sampler_type == 'weighted':
        counts = Counter(domain_labels)
        weights = [1.0 / counts[label] for label in domain_labels]
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    else:
        raise ValueError(f"Unsupported sampler type: {args.sampler}")
