from pathlib import Path
import SimpleITK as sitk
import numpy as np
import torch
import json
from typing import Dict, Tuple
from monai.data.meta_tensor import MetaTensor
from nibabel import orientations as nio

# torch.cuda.set_per_process_memory_fraction(0.2, device=1) # 0.2 * 80GB = 16GB limit; 0.19 * 80 =  15.2 GB safety margin

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

from src.config import Args
from src.inference_ram import run_inference
# from src.inference import run_inference
from src.transforms import Transforms

def get_default_device():
    """Set device for computation"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def sitk_to_monai_dict(img: sitk.Image, key: str = "image"):
    """
    Convert a SimpleITK Image into a MONAI MetaTensor-style dict
    suitable for dictionary-based transforms (e.g., OrientationD, SpacingD).
    """

    # Extract numpy array [D, H, W]
    array = sitk.GetArrayFromImage(img)
    # print("Original sitk array shape:", array.shape)
    array = np.transpose(array, (2, 1, 0))  #  -> (W, H, D)
    affine = np.eye(4, dtype=np.float32)
    affine[0,0] = -0.3
    affine[1,1] = -0.3
    affine[2,2] = 0.3
    
    tensor = MetaTensor(
        torch.as_tensor(array, dtype=torch.float32),
        affine=torch.as_tensor(affine, dtype=torch.float32)
    )
    return {key: tensor}

class ToothFairy3_OralPharyngealSegmentation(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            input_path=Path('/input/images/cbct/'),
            output_path=Path('/output/images/oral-pharyngeal-segmentation/'),
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
                
        # Create output directory if it doesn't exist
        if not self._output_path.exists():
            self._output_path.mkdir(parents=True)
        
        # Create metadata output directory
        self.metadata_output_path = Path('/output/metadata/')
        if not self.metadata_output_path.exists():
            self.metadata_output_path.mkdir(parents=True)
        
        # Initialize device
        self.device = get_default_device() # HERE CHANGE DEVICE
        # self.device = torch.device('cuda:1')
        self.args = Args()
        self.transform = Transforms(self.args, device=self.device)
        
        print(f"Using device: {self.device}")
        print(f"Input path: {self._input_path}")
        print(f"Output path: {self._output_path}")

    def save_instance_metadata(self, metadata: Dict, image_name: str):
        """
        Save instance metadata to JSON file
        
        Args:
            metadata: Instance metadata dictionary
            image_name: Name of the input image (without extension)
        """
        metadata_file = self.metadata_output_path / f"{image_name}_instances.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    @torch.no_grad()
    def predict(self, *, input_image: sitk.Image) -> sitk.Image:
        # print('starting predict method')
        input_tensor = sitk_to_monai_dict(input_image, key=self.args.key)
        
        # budget_ram = 12*1024**3
        print(input_tensor["image"].shape)
        # # max_voxels = budget_ram // (48 * 2) #  8GiB = ~90mln pikseli float16, ~512x512x265 = 70 mln pikseli
        # # H, W, D = input_tensor["image"].shape
        # # z_slab_max = max_voxels // (H * W)
        # # # print(z_slab_max)
        # # pz = self.args.patch_size[2]
        # H, W, D = input_tensor["image"].shape
        # memory_needs = H * W * D * 48 * 2
        # if budget_ram < memory_needs:
        #     print("Scan too big, down-sampling to 0.35 mm/px.")
        #     self.transform.inference_preprocessing.transforms[2].spacing_transform.pixdim = np.array([0.35,0.35,0.35])

        # budget_ram = 6.0 * 1024**3   # GiB
        # channels = 48
        # bytes_per_voxel = 2         # float16

        # H, W, D = input_tensor["image"].shape
        # current_voxels = H * W * D
        # max_voxels = budget_ram // (channels * bytes_per_voxel)

        # if current_voxels > max_voxels:
        #     scale_factor = (max_voxels / current_voxels) ** (1/3)
        #     old_pixdim = 0.3
        #     new_pixdim = old_pixdim / scale_factor
        #     print(f"Scan too big, down-sampling with scale {scale_factor:.3f}, new pixdim = {new_pixdim}")
        #     self.transform.inference_preprocessing.transforms[2].spacing_transform.pixdim = new_pixdim
        # else:
        #     print("Scan fits in memory, keeping original resolution.")

        input_tensor = self.transform.inference_preprocessing(input_tensor)
        input_tensor["image"] = input_tensor["image"].unsqueeze(0) # [1, 1, H, W, D] add batch dimension, ass data_loader collate
        print(f"input image size: {input_tensor['image'].shape}, device: {input_tensor['image'].device}, dtype: {input_tensor['image'].dtype}.")
        
        #TEST MAX MEMORY CONSUMPTION
        # input_tensor["image"] = MetaTensor(torch.rand(size = (1,1,768,482,326), device=self.device, dtype=torch.float16))

        output_array = run_inference(input_tensor, self.args, self.device, self.transform)
        # print(output_array.shape)
        #invert to match original
        output_array = np.transpose(output_array, (2, 1, 0))

        print(f"Output shape: {output_array.shape}")
        print(f"Output details: {output_array.max()}, {output_array.min()}")
        print(f"Output details: {output_array.dtype}")

        output_image = sitk.GetImageFromArray(output_array)
        output_image.CopyInformation(input_image)
        
        return output_image


if __name__ == "__main__":
    ToothFairy3_OralPharyngealSegmentation().process()
