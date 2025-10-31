#!/usr/bin/env python3
import os
import numpy as np
from itertools import product
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import label, find_objects
from skimage.morphology import skeletonize
import networkx as nx

# ---- CONFIG ----
INPUT_DIR = "data/labelsTr"
OUTPUT_DIR = "data/labelsRemapped"
NUM_CHANNELS = "all"  # 1 for primary only, 2 for primary + pulp, "all" for both

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- LABEL MAPPINGS ----
primary_mapping = {
    0: 0,  # background
    1: 1,  # lower jawbone
    2: 2,  # upper jawbone
    3: 3,  # left inf. alveolar canal
    4: 4,  # right inf. alveolar canal
    5: 5,  # left maxillary sinus
    6: 6,  # right maxillary sinus
    7: 7,  # pharynx
    8: 8,  # bridge
    9: 9,  # crown
    10: 10,  # implant
    # Teeth (32)
    11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18,
    21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 26: 24, 27: 25, 28: 26,
    31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32, 37: 33, 38: 34,
    41: 35, 42: 36, 43: 37, 44: 38, 45: 39, 46: 40, 47: 41, 48: 42,
}

PULP_IDS = {
    111, 112, 113, 114, 115, 116, 117, 118,
    121, 122, 123, 124, 125, 126, 127, 128,
    131, 132, 133, 134, 135, 136, 137, 138,
    141, 142, 143, 144, 145, 146, 147, 148
}

def bridge_pulp_graph(pulp_mask, primary, max_gap=2):
    """
    Connect small discontinuities in pulp per tooth using skeletonization and graph bridging.

    Args:
        pulp_mask: np.uint8 array (1 where pulp exists)
        primary: int array of same shape (tooth IDs)
        max_gap: maximum number of voxels to bridge along skeleton paths

    Returns:
        np.uint8 array with gaps bridged
    """
    pulp_fixed = np.copy(pulp_mask)
    tooth_ids = np.unique(primary)
    tooth_ids = [tid for tid in tooth_ids if 11 <= tid <= 48]

    for tid in tooth_ids:
        # Tooth ROI
        mask_tooth = (primary == tid)
        if mask_tooth.sum() == 0:
            continue
        slices = find_objects(mask_tooth)
        if not slices or slices[0] is None:
            continue
        roi_slicer = slices[0]

        pulp_roi = pulp_mask[roi_slicer]
        tooth_roi = mask_tooth[roi_slicer]

        # Skeletonize pulp within tooth
        skeleton = skeletonize(pulp_roi.astype(bool))

        # Build graph from skeleton voxels
        coords = np.argwhere(skeleton)
        G = nx.Graph()
        for idx, (z, y, x) in enumerate(coords):
            G.add_node(idx, coord=(z, y, x))
        # Connect 6-neighbors in skeleton
        coord_to_idx = {tuple(c): i for i, c in enumerate(coords)}
        for i, (z, y, x) in enumerate(coords):
            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if abs(dz) + abs(dy) + abs(dx) != 1:
                            continue
                        neighbor = (z+dz, y+dy, x+dx)
                        if neighbor in coord_to_idx:
                            G.add_edge(i, coord_to_idx[neighbor])

        # Find connected components of skeleton
        components = list(nx.connected_components(G))
        # Compare pairwise components to bridge small gaps
        for i, comp_a in enumerate(components):
            for j, comp_b in enumerate(components):
                if i >= j:
                    continue
                # minimum voxel distance between two components
                coords_a = np.array([G.nodes[n]['coord'] for n in comp_a])
                coords_b = np.array([G.nodes[n]['coord'] for n in comp_b])
                dists = np.sum((coords_a[:, None, :] - coords_b[None, :, :])**2, axis=2)
                min_dist = np.sqrt(dists.min())
                if min_dist <= max_gap:
                    # Bridge: set line of voxels between nearest points
                    idx_a, idx_b = np.unravel_index(dists.argmin(), dists.shape)
                    z0, y0, x0 = coords_a[idx_a]
                    z1, y1, x1 = coords_b[idx_b]
                    # Simple Bresenham-like line
                    num_points = int(max(abs(z1-z0), abs(y1-y0), abs(x1-x0))) + 1
                    for t in np.linspace(0, 1, num_points):
                        zi = int(round(z0*(1-t) + z1*t))
                        yi = int(round(y0*(1-t) + y1*t))
                        xi = int(round(x0*(1-t) + x1*t))
                        pulp_roi[zi, yi, xi] = 1

        # Assign back
        pulp_fixed[roi_slicer] = pulp_roi

    return pulp_fixed.astype(np.uint8)

# ---- REMAP FUNCTION ----
def remap_labels(data):
    primary = np.copy(data)
    pulp_mask = np.zeros_like(data, dtype=np.uint8)

    # Iterate through all pulp classes
    for pulp_id in PULP_IDS:
        tooth_id = pulp_id % 100  # last two digits are the tooth ID

        # Mark pulp in secondary channel
        pulp_mask[data == pulp_id] = 1

        # Replace pulp voxels with tooth class in primary channel
        primary[data == pulp_id] = tooth_id
        
    # pulp_mask = bridge_pulp_graph(
    #     pulp_mask=pulp_mask,
    #     primary=primary,
    #     max_gap=3
    # )
    
    return primary, pulp_mask

def remap_labels_fast(data):
    data = data.astype(np.int32, copy=False)

    # Create mapping from ID → tooth ID (or itself if not pulp) - i.e. change pulp IDs to tooth IDs
    pulp_ids = np.array(list(PULP_IDS), dtype=np.int32)
    max_id = max(data.max(), pulp_ids.max())
    mapping = np.arange(max_id + 1, dtype=np.int32)
    tooth_ids = pulp_ids % 100
    mapping[pulp_ids] = tooth_ids

    # Apply mapping to get primary labels
    primary = mapping[data]

    # Create pulp mask in one shot - all pulp IDs become 1, rest 0
    pulp_mask = np.isin(data, pulp_ids).astype(np.uint8)
    
    return primary, pulp_mask

# ---- PROCESS ALL FILES ----
label_files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".nii.gz"))

for fname in tqdm(label_files, desc="Remapping labels"):
    path_in = os.path.join(INPUT_DIR, fname)
    img = nib.load(path_in)
    data = np.asarray(img.dataobj, dtype=np.int32)

    primary, pulp = remap_labels_fast(data)
    
    if NUM_CHANNELS == 1 or NUM_CHANNELS == "all":
        # Save primary
        img_primary = nib.Nifti1Image(primary.astype(np.uint8), affine=img.affine, header=img.header)
        nib.save(img_primary, os.path.join(OUTPUT_DIR, fname.replace(".nii.gz", "_primary.nii.gz")))

        # Save pulp
        img_pulp = nib.Nifti1Image(pulp.astype(np.uint8), affine=img.affine, header=img.header)
        nib.save(img_pulp, os.path.join(OUTPUT_DIR, fname.replace(".nii.gz", "_pulp.nii.gz")))
        
    if NUM_CHANNELS == 2 or NUM_CHANNELS == "all":
        # Combine into multi-channel array
        multi_channel = np.stack([primary.astype(np.uint8), pulp.astype(np.uint8)], axis=-1)

        # Save as single NIfTI
        hdr = img.header.copy()        # copy to avoid modifying original
        hdr.set_data_dtype(np.uint8)
        hdr.set_data_shape(multi_channel.shape)
        hdr['dim'][0] = 4              # Now it's 4D (X,Y,Z,channels)
        hdr['dim'][4] = 2              # Two channels
        hdr['intent_code'] = 1007      # NIFTI_INTENT_VECTOR
        hdr['intent_name'] = b'Vector'
        img_multi = nib.Nifti1Image(multi_channel, affine=img.affine, header=hdr)
        nib.save(img_multi, os.path.join(OUTPUT_DIR, fname))
    
print("Remapping complete. Saved to", OUTPUT_DIR)