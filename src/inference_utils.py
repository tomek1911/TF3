import os
import torch
import numpy as np
from monai.transforms import SaveImage

original_to_index_map = {
    0: 0,   1: 1,  2: 2,  3: 3,  4: 4,  5: 5,  6: 6,  7: 7,  8: 8,  9: 9,  10: 10,
    11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18,
    21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 26: 24, 27: 25, 28: 26,
    31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32, 37: 33, 38: 34,
    41: 35, 42: 36, 43: 37, 44: 38, 45: 39, 46: 40, 47: 41, 48: 42,
    103: 43, 104: 44, 105: 45,
}

def save_inference_multiclass_segmentation(
    output_dir: str,
    array: np.ndarray,
    meta_dict: dict,
    invert_transform,
    original_image,
    name: str,
    is_invert_mapping: bool = True,
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