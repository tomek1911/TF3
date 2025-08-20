from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Args:
    patch_size: Tuple[int, int, int] = (288, 288, 160)
    pixdim: float = 0.3
    houndsfield_clip: int = 3000
    key: str = "image"
    n_features: int = 32
    unet_depth: int = 5
    norm: str = "instance"
    activation: str = "relu"
    backbone_name: str = "resnet34"
    out_channels: int = 46
    classes: int = 45
    configuration: str = "DIST_PULP"
    inference_autocast_dtype: str = "float16"