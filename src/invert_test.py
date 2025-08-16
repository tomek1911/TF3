import torch
from monai.transforms import (
    Compose,
    SpatialPadd,
    Spacingd,
    Invertd,
    EnsureTyped,
)
from monai.data import MetaTensor

# Create a simple tensor image (e.g., 1 channel, 3x3x3)
image = torch.arange(27, dtype=torch.float32).reshape(1, 3, 3, 3)
meta_image = {"image": MetaTensor(image, affine=torch.eye(4))}

# Define deterministic transforms
transform = Compose([
    SpatialPadd(keys="image", spatial_size=(5, 5, 5)),
    Spacingd(keys="image", pixdim=(0.5, 0.5, 0.5), mode="bilinear"),
    EnsureTyped(keys="image", dtype=torch.float32),
])

# Apply transforms
transformed = transform(meta_image)

# Invert the transforms
invert_transform = Invertd(
    keys="pred",
    transform=transform,
    orig_keys="image",
    to_tensor=True
)

# Simulate prediction dictionary
pred_dict = {"pred": transformed["image"], "image": meta_image["image"]}

# Apply inverse
inverted = invert_transform(pred_dict)

print("Original shape:", image.shape)
print("Transformed shape:", transformed["image"].shape)
print("Inverted shape:", inverted["pred"].shape)
print("Inverted tensor equals original:", torch.allclose(inverted["pred"], image))
