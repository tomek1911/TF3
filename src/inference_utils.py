import os
import torch
import numpy as np
import nibabel as nib
from monai.data import MetaTensor, decollate_batch
from monai.transforms import Orientationd
from monai.transforms import SaveImage

original_to_index_map = {
    0: 0,   1: 1,  2: 2,  3: 3,  4: 4,  5: 5,  6: 6,  7: 7,  8: 8,  9: 9,  10: 10,
    11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18,
    21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 26: 24, 27: 25, 28: 26,
    31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32, 37: 33, 38: 34,
    41: 35, 42: 36, 43: 37, 44: 38, 45: 39, 46: 40, 47: 41, 48: 42,
    103: 43, 104: 44, 105: 45,
}

pred_to_challange_map = {
    0: 0,  1: 1,  2: 2,  3: 3, 4: 4,  5: 5, 6: 6,  7: 7,  8: 8,  9: 9, 10: 10,
    11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18,
    19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 26, 25: 27, 26: 28,
    27: 31, 28: 32, 29: 33, 30: 34, 31: 35, 32: 36, 33: 37, 34: 38,
    35: 41, 36: 42, 37: 43, 38: 44, 39: 45, 40: 46, 41: 47, 42: 48,
    43: 51, 44: 52, 45: 53, 46: 50 #pulp 50 stays the same
}

def save_nifti(array, path, filename, pixdim = 0.4, label_meta_dict=None, affine=None, dtype = np.int16):
    if label_meta_dict is None:
        affine = np.eye(4) * pixdim
        affine[3][3]=1.0
    else:
        if len(array.shape)==5:
            label_meta_dict = decollate_batch(label_meta_dict)[0]
            array = array[0]
        affine = label_meta_dict["affine"].numpy()
        space = label_meta_dict['space']
        if nib.aff2axcodes(affine) == ('L', 'P', 'S') and space == "RAS":
            t = MetaTensor(array, meta=label_meta_dict)
            array=Orientationd(keys="label", axcodes="RAS")({"label": t})["label"]
    if torch.is_tensor(array):
        nib_array = nib.Nifti1Image(array.cpu().squeeze().numpy().astype(dtype), affine=affine)
    else:
        nib_array = nib.Nifti1Image(array.astype(dtype), affine=affine)
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = os.path.join(path, filename)
    nib.save(nib_array, save_path)

def save_inference_multiclass_segmentation(
    output_dir: str,
    array: np.ndarray,
    meta_dict: dict,
    invert_transform,
    original_image,
    name: str,
    is_invert_mapping: bool = False,
    label_map: dict = None,
    dtype=np.float32
):
    """
    Save predicted outputs (numpy array) as NIfTI, restoring original labels and geometry.

    Args:
        output_dir: Directory where the file will be saved.
        array: Prediction array (numpy).
        meta_dict: Metadata from the original tensor/image (affine, spacing, filename).
        invert_transform: MONAI invert transform to restore spacing/origin.
        original_image: Original image tensor used in inference.
        name: Postfix for the saved file.
        is_invert_mapping: Whether to restore labels to original IDs.
        label_map: Mapping from internal label IDs to original label IDs.
        dtype: Data type for saving.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Invert preprocessing transforms (restore spacing/origin)
    data_dict = {"pred": array, "image": original_image}
    pred_nifti = invert_transform(data_dict)["pred"]

    # Restore original labels if required
    if is_invert_mapping:
        if label_map is None:
            # invert mapping from original_to_index_map
            label_map = {v: k for k, v in original_to_index_map.items()}
        restored_pred = np.zeros_like(pred_nifti, dtype=np.uint16)
        for idx_val, orig_val in label_map.items():
            restored_pred[pred_nifti == idx_val] = orig_val
        pred_nifti = restored_pred

    # Save image using MONAI SaveImage
    saver = SaveImage(
        output_dir=output_dir,
        output_postfix=f"_{name}",
        output_ext=".nii.gz",
        dtype=dtype,
        resample=False  # keep original geometry
    )
    saver(pred_nifti, meta_dict)
    
def save_float_map(
    output_dir: str,
    array: np.ndarray,
    meta_dict: dict,
    invert_transform,
    original_image,
    name: str,
    dtype=np.float32
):
    """
    Save continuous-valued maps (float or binary) as NIfTI after inversion.

    Args:
        output_dir: Directory to save the file.
        array: Map array (numpy) to save.
        meta_dict: Metadata from the original tensor/image (affine, spacing, filename).
        invert_transform: MONAI invert transform to restore spacing/origin.
        original_image: Original image tensor used in preprocessing.
        name: Postfix for the saved file.
        dtype: Data type for saving.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Invert preprocessing transforms (restore spacing/origin)
    data_dict = {"pred": array, "image": original_image}
    map_nifti = invert_transform(data_dict)["pred"]

    # Save map using MONAI SaveImage
    saver = SaveImage(
        output_dir=output_dir,
        output_postfix=f"_{name}",
        output_ext=".nii.gz",
        dtype=dtype,
        resample=False  # preserve original geometry
    )
    saver(map_nifti, meta_dict)
    
def merge_pulp_into_teeth(multiclass_pred: np.ndarray, 
                          pulp_pred: np.ndarray, 
                          pulp_class: int = 50, 
                          excluded_classes=None) -> np.ndarray:
    """
    Merge pulp segmentation into teeth predictions.
    
    Args:
        teeth_pred: 3D array of teeth instance predictions (e.g., from deep watershed)
        pulp_pred: 3D array of binary pulp segmentation (1 = pulp, 0 = background)
        pulp_class: integer label to assign to pulp voxels inside teeth (default=111)
        excluded_classes: list of class labels to exclude from teeth mask
    
    Returns:
        merged_pred: 3D array with pulp merged into teeth, respecting tooth mask
    """
    if excluded_classes is None:
        excluded_classes = list(range(0, 11)) + [43, 44, 45]
        
    merged_pred = multiclass_pred.copy()
    
    # Mask: valid teeth voxels (not in excluded classes)
    teeth_mask = ~np.isin(multiclass_pred, excluded_classes)
    
    # Only keep pulp inside valid teeth
    pulp_inside_teeth = (pulp_pred > 0) & teeth_mask
    
    # Assign pulp class
    merged_pred[pulp_inside_teeth] = pulp_class
    
    return merged_pred

def merge_pulp_into_teeth_torch(
    multiclass_pred: torch.Tensor,
    pulp_pred: torch.Tensor,
    pulp_class: int = 50,
    excluded_classes = None,
) -> torch.Tensor:
    """
    Merge pulp segmentation into teeth predictions using PyTorch tensors.
    
    Args:
        multiclass_pred: 3D tensor of teeth instance predictions (H, W, D)
        pulp_pred: 3D tensor of binary pulp segmentation (1 = pulp, 0 = background)
        pulp_class: integer label to assign to pulp voxels inside teeth
        excluded_classes: list of class labels to exclude from teeth mask
    
    Returns:
        merged_pred: 3D tensor with pulp merged into teeth, respecting tooth mask
    """
    if excluded_classes is None:
        excluded_classes = list(range(0, 11)) + [43, 44, 45] # do not modify non-teeth classes
    
    #merged_pred = multiclass_pred.clone() - no need to clone, we will modify in place
    
    # Mask: valid teeth voxels (not in excluded classes)
    excluded_tensor = torch.tensor(excluded_classes, device=multiclass_pred.device)
    teeth_mask = ~torch.isin(multiclass_pred, excluded_tensor)
    
    # Only keep pulp inside valid teeth
    pulp_inside_teeth = (pulp_pred > 0) & teeth_mask
    
    # Assign pulp class
    multiclass_pred[pulp_inside_teeth] = pulp_class
    
    return multiclass_pred

def remap_labels_torch(seg_pred: torch.Tensor, mapping_dict: dict) -> torch.Tensor:
    """
    Remap segmentation labels using a lookup table on GPU/CPU with PyTorch.
    
    Args:
        seg_pred: torch.Tensor of integer labels (any shape).
        mapping_dict: dict mapping original labels -> new labels.
    
    Returns:
        torch.Tensor with remapped labels, same shape as input.
    """
    max_key = max(mapping_dict.keys())
    lut = torch.full((max_key + 1,), -1, dtype=torch.int32, device=seg_pred.device)
    for k, v in mapping_dict.items():
        lut[k] = v
    return lut[seg_pred]


def remap_labels_numpy(seg_pred: np.ndarray, mapping_dict: dict) -> np.ndarray:
    """
    Remap segmentation labels using a lookup table with NumPy.
    
    Args:
        seg_pred: np.ndarray of integer labels (any shape).
        mapping_dict: dict mapping original labels -> new labels.
    
    Returns:
        np.ndarray with remapped labels, same shape as input.
    """
    max_key = max(mapping_dict.keys())
    lut = np.full((max_key + 1,), -1, dtype=np.int32)
    for k, v in mapping_dict.items():
        lut[k] = v
    return lut[seg_pred]