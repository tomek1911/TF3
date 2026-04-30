import torch

from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    EnsureTypeD,
    InvertD,
    LoadImageD,
    OrientationD,
    ScaleIntensityRangeD,
    SpacingD,
    SpatialPadD,
    ToDeviceD,
)


class Transforms:
    """
    Preprocessing and postprocessing for centerF inference.

    Preprocessing pipeline (applied to a dict with key "image"):
      LoadImageD       → load .nii.gz (or any NiBabel-readable format) from disk
      EnsureChannelFirstD
      OrientationD     → reorient to RAS
      SpacingD         → resample to pixdim (default 0.6 mm isotropic)
      ScaleIntensityRangeD → clip HU [0, houndsfield_clip] → [0.0, 1.0]
      SpatialPadD      → pad to at least patch_size (so the first patch fits)
      EnsureTypeD      → cast to fp16
      ToDeviceD        → move to GPU

    Postprocessing (InvertD):
      Inverts SpacingD / OrientationD / SpatialPadD back to the original
      image geometry.  nearest-neighbour interpolation is used so label
      values are preserved exactly.

    Usage:
        t = Transforms(args, device)
        data = t.preprocess({"image": "/path/to/scan.nii.gz"})
        # data["image"] is a (1, H, W, D) fp16 MetaTensor on device
        data["image"] = data["image"].unsqueeze(0)   # add batch dim → (1,1,H,W,D)
        ...run inference...
        result = t.postprocess({"pred": pred_metatensor, "image": data["image"][0]})
        # result["pred"] is the prediction in the original image geometry
    """

    def __init__(self, args, device: torch.device = torch.device("cpu")):
        pixdim = (args.pixdim,) * 3

        self._preprocessing = Compose([
            LoadImageD(keys="image", reader="NibabelReader"),
            EnsureChannelFirstD(keys="image"),
            OrientationD(keys="image", axcodes="RAS"),
            SpacingD(keys="image", pixdim=pixdim, mode="bilinear"),
            ScaleIntensityRangeD(
                keys="image",
                a_min=0, a_max=args.houndsfield_clip,
                b_min=0.0, b_max=1.0,
                clip=True,
            ),
            SpatialPadD(
                keys="image",
                spatial_size=args.patch_size,
                mode="constant",
                constant_values=0,
            ),
            EnsureTypeD(keys="image", dtype=torch.float16),
            ToDeviceD(keys="image", device=device),
        ])

        self._postprocessing = Compose([
            InvertD(
                keys="pred",
                transform=self._preprocessing,
                orig_keys="image",
                nearest_interp=True,
                to_tensor=True,
            ),
        ])

    def preprocess(self, data: dict) -> dict:
        """
        Args:
            data: {"image": <path string>}
        Returns:
            dict with data["image"] as (1, H, W, D) fp16 MetaTensor on device.
        """
        return self._preprocessing(data)

    def postprocess(self, data: dict) -> dict:
        """
        Args:
            data: {"pred": MetaTensor (1, H, W, D), "image": original preprocessed image}
        Returns:
            dict with data["pred"] restored to original image geometry.
        """
        return self._postprocessing(data)
