import SimpleITK as sitk
import os
import numpy as np
import nibabel as nib

# p_id = "F_001" #P_077, P_381, S_0042, F_001
# p = f"data/labelsTr/ToothFairy3{p_id}.nii.gz"
# p_out = f"test-data/labelsTr/ToothFairy3{p_id}.mha"

# label = nib.load(p)
# mask_data = label.get_fdata()

# mask_sitk = sitk.GetImageFromArray(mask_data)
# writer = sitk.ImageFileWriter()
# writer.SetFileName(p_out)
# writer.Execute(mask_sitk)

p_id = "F_001"
p = f"data/labelsTr/ToothFairy3{p_id}.nii.gz"
p_out = f"test-data/labelsTr/ToothFairy3{p_id}.mha"

# load with nibabel (voxel data + affine)
label = nib.load(p)
mask_data = label.get_fdata()
affine = label.affine

# convert to sitk
mask_sitk = sitk.GetImageFromArray(mask_data.astype(np.int16))  # or float, depending on your labels

# extract spacing, origin, direction from affine
# nibabel affine: maps voxel indices to world coords
spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
direction = (affine[:3, :3] / spacing).flatten()
origin = affine[:3, 3]

mask_sitk.SetSpacing(tuple(spacing))
mask_sitk.SetDirection(tuple(direction))
mask_sitk.SetOrigin(tuple(origin))

# save
sitk.WriteImage(mask_sitk, p_out)