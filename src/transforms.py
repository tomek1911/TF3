import torch
import numpy as np
from monai.transforms import (
    AsDiscrete,
    Activations,
    Compose,
    ToDevice
)

from monai.transforms import (
    EnsureChannelFirstD,
    CopyItemsD,
    EnsureTypeD,
    LambdaD,
    LoadImageD,
    OrientationD,
    ResizeD,
    RandAdjustContrastD,
    RandAffineD,
    RandLambdaD,
    RandRotateD,
    RandSpatialCropD,
    RandScaleIntensityD,
    RandShiftIntensityD,
    RandSpatialCropSamplesD,
    RandZoomD,
    RandFlipD,
    RandRotate90D,
    ResizeWithPadOrCropD,
    ScaleIntensityRangeD,
    SpacingD,
    ThresholdIntensityD,
    ToDeviceD,
)


class Transforms():
    def __init__(self,
                 args,
                 device: torch.device = torch.device("cpu")
                 ) -> None:

        self.pixdim = (args.pixdim,)*3
        self.keys = args.keys
        mode_interpolation_dict = {"image":   "bilinear",
                                   "label":   "nearest",
                                   "binary_label":   "nearest",
                                   "edt":     "bilinear",
                                   "edt_dir": "bilinear",
                                   }
        self.mode = [mode_interpolation_dict[key] for key in self.keys]

    
        self.preprocessing_transforms = [
            LoadImageD(keys=self.keys, reader='NibabelReader'),
            EnsureChannelFirstD(keys=self.keys, channel_dim='no_channel'),
            OrientationD(keys=self.keys, axcodes="RAS"),
            ToDeviceD(keys=self.keys, device=device),
            EnsureTypeD(keys=self.keys, data_type="tensor"),
            SpacingD(keys=self.keys, pixdim=self.pixdim, mode=self.mode),
            ScaleIntensityRangeD(keys="image", a_min=0, a_max=args.houndsfield_clip, b_min=0.0, b_max=1.0, clip=True)]
        self.geometric_transforms = []
        self.intensity_transforms = []
        
        self.train_transform = Compose(self.preprocessing_transforms + self.geometric_transforms +
                                       self.intensity_transforms, lazy=args.lazy_interpolation)
        self.inference_transform = Compose(self.preprocessing_transforms)

        self.post_pred_train = Compose([Activations(softmax=True, dim=1),
                                        AsDiscrete(argmax=True,
                                                   dim=1,
                                                   keepdim=True)
                                        ])
        self.post_pred_inference = Compose([Activations(softmax=True, dim=0),
                                            AsDiscrete(argmax=True,
                                                       dim=0,
                                                       keepdim=True)
                                            ])
