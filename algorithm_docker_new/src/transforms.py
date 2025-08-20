import torch

from monai.transforms import (
    ActivationsD,
    AsDiscreteD,
    Compose,
    EnsureChannelFirstD,
    EnsureTypeD,
    InvertD,
    OrientationD,
    ScaleIntensityRangeD,
    SpacingD,
    SpatialPadD,
    ToDeviceD,
)

class Transforms():
    def __init__(self,
                 args,
                 device: torch.device = torch.device("cpu")
                 ) -> None:

        self.pixdim = (args.pixdim,)*3
        self.preprocessing_inference_transforms = [
            EnsureChannelFirstD(keys="image", channel_dim='no_channel'),
            OrientationD(keys="image", axcodes="RAS"), #axcodes RAS is a target, which must match affine from file
            SpacingD(keys="image", pixdim=self.pixdim, mode="bilinear"),
            ScaleIntensityRangeD(keys="image", a_min=0, a_max=args.houndsfield_clip, b_min=0.0, b_max=1.0, clip=True),
            SpatialPadD(keys="image", spatial_size=args.patch_size, mode="constant", constant_values=0),
            EnsureTypeD(keys="image", dtype=torch.float16),
            ToDeviceD(keys="image", device=device)
            ]
    
        self.inference_preprocessing = Compose(self.preprocessing_inference_transforms)
        
        self.post_inference_transform = Compose(
                [
                    InvertD(
                        keys='pred',
                        transform= self.inference_preprocessing,
                        orig_keys="image",
                        nearest_interp=False,
                        to_tensor=True
                    ),
                    # AsDiscreteD(keys="pred", threshold=0.5)
                ]
            )