import torch
import numpy as np
from functools import partial

from monai.transforms import (
    AsDiscrete,
    Activations,
    Compose,
    ToDevice
)

from monai.transforms import (
    ActivationsD,
    AsDiscreteD,
    CopyItemsD,
    EnsureChannelFirstD,
    EnsureTypeD,
    InvertD,
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
    ResizeWithPadOrCropD,
    SaveImageD,
    ScaleIntensityRangeD,
    SpacingD,
    SpatialPadD,
    ThresholdIntensityD,
    ToDeviceD,
)
original_to_index_map = {
    0: 0,   1: 1,  2: 2,  3: 3,  4: 4,  5: 5,  6: 6,  7: 7,  8: 8,  9: 9,  10: 10,
    11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18,
    21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 26: 24, 27: 25, 28: 26,
    31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32, 37: 33, 38: 34,
    41: 35, 42: 36, 43: 37, 44: 38, 45: 39, 46: 40, 47: 41, 48: 42,
    103: 43, 104: 44, 105: 45
}
class SaveMultipleKeysD:
    def __init__(self, keys, output_dir, output_postfixes, separate_folder, output_dtype=None):
        self.keys = keys
        self.postfixes = output_postfixes
        self.output_dir = output_dir
        self.separate_folder = separate_folder
        self.output_dtype = output_dtype
        
    def __call__(self, data):
        for idx, (key, postfix) in enumerate(zip(self.keys, self.postfixes)):
            dtype = self.output_dtype[idx] if self.output_dtype is not None else None
            SaveImageD(
                keys=[key],
                output_dir=self.output_dir,
                output_postfix=postfix,
                separate_folder=self.separate_folder,
                output_dtype=dtype
            )(data)
        return data

def move_to_device(x, device):
    if x.device == device:
        return x
    else:
        return x.to(device, non_blocking=True)
    
def to_float(x):
    return x.float()

def to_long(x):
    return x.long()

def remap_labels(label: torch.Tensor, mapping: dict, channel_id: int = 0, default_class: int = 0) -> torch.Tensor:
    """
    Map arbitrary class labels to contiguous indices for a specific channel.

    Args:
        label: [B,C,H,W,D] or [C,H,W,D] tensor of original labels
        mapping: dict {original_label: new_index}
        channel_id: which channel to remap

    Returns:
        label_indexed: same shape as label, with remapped channel
    """
    # Make a copy to avoid modifying original tensor
    label_indexed = label.clone()

    # select the channel to remap
    channel_data = label_indexed[:, channel_id:channel_id+1] if label.dim() == 5 else label_indexed[channel_id:channel_id+1]
    
    # replace values that are truly unknown
    unknown_values = [v for v in torch.unique(channel_data).cpu().numpy() if v not in mapping]
    is_mask_unknown = False
    if unknown_values:
        # print (f"Unknown label values found: {unknown_values}, file: {label.meta['filename_or_obj']}")
        is_mask_unknown = True

    # perform mapping
    orig_values = torch.tensor(sorted(mapping.keys()), device=label.device).contiguous()
    new_indices = torch.tensor([mapping[v.item()] for v in orig_values], device=label.device)
    flat_label = channel_data.flatten().contiguous()
    if is_mask_unknown:
        mask_unknown = ~torch.isin(flat_label, orig_values)
        flat_label[mask_unknown] = default_class

    idx = torch.bucketize(flat_label, orig_values)
    flat_mapped = new_indices[idx]
    mapped_channel = flat_mapped.view(channel_data.shape)

    # replace original channel with mapped channel
    if label.dim() == 5:
        label_indexed[:, channel_id:channel_id+1] = mapped_channel
    else:
        label_indexed[channel_id:channel_id+1] = mapped_channel

    return label_indexed
class Transforms():
    def __init__(self,
                 args,
                 device: torch.device = torch.device("cpu")
                 ) -> None:

        self.pixdim = (args.pixdim,)*3
        self.keys = args.keys
        mode_interpolation_dict = {"image":   "bilinear",
                                   "label":   "nearest",
                                   "watershed_map":  "bilinear",
                                   }
        if args.use_anisotropic:
            min_zoom = (args.min_zoom,) * 3
            max_zoom = (args.max_zoom,) * 3
        else:
            min_zoom = args.min_zoom
            max_zoom = args.max_zoom
        self.mode = [mode_interpolation_dict[key] for key in self.keys]
        
        self.preprocessing_transforms = [
            LoadImageD(keys=self.keys, reader='NibabelReader'),
            EnsureChannelFirstD(keys=self.keys),
            OrientationD(keys=self.keys, axcodes="RAS"),
            SpacingD(keys=self.keys, pixdim=self.pixdim, mode=self.mode),
            ScaleIntensityRangeD(keys="image", a_min=0, a_max=args.houndsfield_clip, b_min=0.0, b_max=1.0, clip=True),
            SpatialPadD(keys=self.keys, spatial_size=args.patch_size, mode="constant", constant_values=0),
            LambdaD(keys="label", func=partial(remap_labels, mapping=original_to_index_map, channel_id=0))
            ]
        
        self.preprocessing_inference_transforms = [
            LoadImageD(keys="image", reader='NibabelReader'),
            EnsureChannelFirstD(keys="image"),
            OrientationD(keys="image", axcodes="RAS"),
            SpacingD(keys="image", pixdim=self.pixdim, mode="bilinear"),
            ScaleIntensityRangeD(keys="image", a_min=0, a_max=args.houndsfield_clip, b_min=0.0, b_max=1.0, clip=True),
            SpatialPadD(keys="image", spatial_size=args.patch_size, mode="constant", constant_values=0),
            EnsureTypeD(keys="image", dtype=torch.float16),
            ToDeviceD(keys="image", device=device)
            ]
        
        self.pre_collate = [
            RandSpatialCropSamplesD(keys=self.keys, roi_size=args.patch_size, random_size=False, num_samples=args.crop_samples)
        ]
               
        self.geometric_transforms = [
            RandLambdaD(keys=self.keys, prob=1.0, func=partial(move_to_device, device=device)), # make sure that data are on the correct gpu
            RandRotateD(keys=self.keys, range_x=args.rotation_range_xy, range_y=args.rotation_range_xy, range_z=args.rotation_range_z, prob=args.rotation_prob, mode=self.mode),
            RandZoomD(keys=self.keys, min_zoom=min_zoom, max_zoom=max_zoom, prob=args.zoom_prob, mode=self.mode) #anisotropic zoom
        ]

        self.intensity_transforms = [
            RandScaleIntensityD(keys="image", factors=args.intensity_scale, prob=args.intensity_scale_prob),
            RandShiftIntensityD(keys="image", offsets=args.intensity_shift, prob=args.intensity_shift_prob),
            RandAdjustContrastD(keys="image", gamma=(1.0 - args.contrast, 1.0 + args.contrast), prob=args.intensity_contrast_prob)
        ]

        self.final_transform = [
            LambdaD(keys=["image", "watershed_map"], func=to_float),
            LambdaD(keys="label", func=to_long)
        ]
        if not args.use_augmentations:
            self.geometric_transforms = []
            self.intensity_transforms = []
            
        self.train_transform = Compose(self.preprocessing_transforms + self.geometric_transforms +
                                       self.intensity_transforms + self.final_transform, 
                                       lazy=getattr(args, "lazy_interpolation", False))
        self.train_transform.set_random_state(seed=args.seed, state=np.random.RandomState(seed=args.seed))
        
        self.pre_collate_transform = Compose(self.pre_collate)
        self.pre_collate_transform.set_random_state(seed=args.seed, state=np.random.RandomState(seed=args.seed))
        
        self.cpu_transform = Compose(self.preprocessing_transforms)
        self.gpu_transform = Compose(self.geometric_transforms + self.intensity_transforms + self.final_transform, lazy=getattr(args, "lazy_interpolation", False))
        self.gpu_transform.set_random_state(seed=args.seed, state=np.random.RandomState(seed=args.seed))
        
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
        self.post_transform_binary = Compose(
                        [
                            ActivationsD(keys="pred", sigmoid=True),
                            InvertD(
                                keys="pred",  # invert the `pred` data field, also support multiple fields
                                transform= self.inference_transform,
                                orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
                                # then invert `pred` based on this information. we can use same info
                                # for multiple fields, also support different orig_keys for different fields
                                nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
                                # to ensure a smooth output, then execute `AsDiscreted` transform
                                to_tensor=True,  # convert to PyTorch Tensor after inverting
                            ),
                            AsDiscreteD(keys="pred", threshold=0.5),
                            SaveImageD(keys="pred", output_dir="output", output_postfix="seg", resample=False, separate_folder=False),
                        ]
                    )
        
        self.inference_preprocessing = Compose(self.preprocessing_inference_transforms)
        
        self.post_inference_transform = Compose(
                [
                    InvertD(
                        keys=['mlt','pulp','dist', 'dir'],  # invert the `pred` data field, also support multiple fields
                        transform= self.inference_preprocessing,
                        orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
                        # then invert `pred` based on this information. we can use same info
                        # for multiple fields, also support different orig_keys for different fields
                        nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
                        # to ensure a smooth output, then execute `AsDiscreted` transform
                        to_tensor=True  # convert to PyTorch Tensor after inverting
                    ),
                    # AsDiscreteD(keys="pred", threshold=0.5),
                    # SaveImageD(keys="pred", output_dir="output", output_postfix="seg", resample=False, separate_folder=False),
                ]
            )
        self.post_inference_transform_no_dir = Compose(
                [
                    InvertD(
                        keys=['mlt','pulp','dist'],  # invert the `pred` data field, also support multiple fields
                        transform= self.inference_preprocessing,
                        orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
                        # then invert `pred` based on this information. we can use same info
                        # for multiple fields, also support different orig_keys for different fields
                        nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
                        # to ensure a smooth output, then execute `AsDiscreted` transform
                        to_tensor=True  # convert to PyTorch Tensor after inverting
                    ),
                    # AsDiscreteD(keys="pred", threshold=0.5),
                    # SaveImageD(keys="pred", output_dir="output", output_postfix="seg", resample=False, separate_folder=False),
                ]
            )
        self.save_inference_output = SaveMultipleKeysD(keys=['mlt','pulp','dist'], output_dir="output", output_postfixes=['mlt','pulp','dist'], separate_folder=False)
        self.save_inference_output_pred = SaveImageD(keys='pred', output_dir="output", output_postfix='pred', separate_folder=False)
        # self.save_inference_output_mlt = SaveImageD(keys="pred", output_dir="output", output_postfix="mlt", resample=False, separate_folder=False)
        # self.save_inference_output_pulp = SaveImageD(keys="pred", output_dir="output", output_postfix="pulp", resample=False, separate_folder=False)
        # self.save_inference_output_dist = SaveImageD(keys="pred", output_dir="output", output_postfix="dist", resample=False, separate_folder=False)
                

if __name__ == "__main__":
    import torch
    import numpy as np
    from monai.data import Dataset, DataLoader

    # -----------------------------
    # Placeholder args
    # -----------------------------
    class Args:
        keys = ["image", "label", "watershed_map"]
        pixdim = 0.3
        houndsfield_clip = 3000
        patch_size = [304, 304, 176]
        crop_samples = 1
        rotation_range_xy = 0.1
        rotation_range_z = 0.15
        rotation_prob = 0.5
        zoom_prob = 0.5
        intensity_scale = 0.15
        intensity_scale_prob = 0.5
        intensity_shift = 0.2
        intensity_shift_prob = 0.5
        contrast = 0.5
        intensity_contrast_prob = 0.5
        min_zoom = 0.95
        max_zoom = 1.05
        zoom_prob = 0.5
        seed = 42
        lazy_interpolation = True
        use_anisotropic = True
        use_augmentations = True

    args = Args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transforms_obj = Transforms(args, device=device)
    sample_data_list = [
        {
            "image": "data/imagesTr/ToothFairy3F_001_0000.nii.gz",
            "label": "data/labelsRemapped/ToothFairy3F_001.nii.gz",
            "watershed_map": "data/deep_watershed_maps/distdir_maps/ToothFairy3F_001.nii.gz"
        },
        {
            "image": "data/imagesTr/ToothFairy3P_001_0000.nii.gz",
            "label": "data/labelsRemapped/ToothFairy3P_001.nii.gz",
            "watershed_map": "data/deep_watershed_maps/distdir_maps/ToothFairy3P_001.nii.gz"
        },
        {
            "image": "data/imagesTr/ToothFairy3S_0001_0000.nii.gz",
            "label": "data/labelsRemapped/ToothFairy3S_0001.nii.gz",
            "watershed_map": "data/deep_watershed_maps/distdir_maps/ToothFairy3S_0001.nii.gz"
        }
    ]
    dataset = Dataset(data=sample_data_list, transform=transforms_obj.cpu_transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for i, batch in enumerate(loader):
        print(f"Sample {i} - {batch['image'].meta['filename_or_obj']}:")
        for key in args.keys:
            data = batch[key]
            print(f"  {key} shape: {data.shape}, dtype: {data.dtype}, device: {data.device}")
            # specific unpacking
            if key == "label":
                primary = data[:, 0, ...]   # channel 0
                pulp = data[:, 1, ...]      # channel 1
                print(f"    Primary label: min={primary.min().item()}, max={primary.max().item()}, unique={torch.unique(primary)}")
                print(f"    Pulp label: min={pulp.min().item()}, max={pulp.max().item()}, unique={torch.unique(pulp)}")

            if key == "watershed_map":
                distance = data[:, 0, ...]  # channel 0
                direction = data[:, 1:, ...]  # channels 1,2,3
                print(f"    Distance map: min={distance.min().item():.4f}, max={distance.max().item():.4f}, mean={distance.mean().item():.4f}")
                print(f"    Direction map: min={direction.min().item():.4f}, max={direction.max().item():.4f}, mean={direction.mean().item():.4f}")    
        