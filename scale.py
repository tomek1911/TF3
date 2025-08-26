import os
import sys
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

def resample_nifti(input_path, scale=(1.5, 1.5, 1.5)):
    # Load scan
    img = nib.load(input_path)
    data = img.get_fdata()
    affine = img.affine.copy()

    # Interpolation
    zoomed_data = zoom(data, zoom=scale, order=3)  # cubic interpolation

    # Update affine matrix (voxel spacing scaling)
    new_affine = affine.copy()
    for i in range(3):
        new_affine[i, i] /= scale[i]

    # Save with new name
    base, ext = os.path.splitext(input_path)
    if ext == ".gz":
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext

    output_path = base + "_big" + ext

    new_img = nib.Nifti1Image(zoomed_data, new_affine, img.header)
    nib.save(new_img, output_path)
    print(f"Saved resampled scan to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python resample_cbct.py input_scan.nii.gz")
        sys.exit(1)

    input_file = sys.argv[1]
    resample_nifti(input_file, scale=(1.5, 1.5, 1.5))