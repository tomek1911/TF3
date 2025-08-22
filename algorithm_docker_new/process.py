from pathlib import Path
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn
import json
from typing import Dict, Tuple
from monai.data.meta_tensor import MetaTensor
from monai.data import decollate_batch
from nibabel import orientations as nio

# torch.cuda.set_per_process_memory_fraction(0.2, device=1) # 0.2 * 80GB = 16GB limit; 0.19 * 80 =  15.2 GB safety margin

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

from src.config import Args
from src.inference import run_inference
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
        # print(input_tensor["image"].meta)
        # print(input_tensor["image"].shape)
        # print(input_tensor["image"].device)
        # input_array = sitk.GetArrayFromImage(input_image) # np.array
        # input_array = np.expand_dims(input_array, axis=0)  # Add batch dimension: [1, H, W, D]
        # print(input_array.shape)
        # input_array = {"image": input_array}
        input_tensor = self.transform.inference_preprocessing(input_tensor)  # preprocessing
        # print(f"applied transforms: {input_tensor['image'].applied_operations}")
        # print(input_tensor["image"].shape)
        input_tensor["image"] = input_tensor["image"].unsqueeze(0) # [1, 1, H, W, D] add batch dimension, ass data_loader collate
        # input_tensor["image"] = torch.rand(size = (1,1,768,482,326), device=self.device, dtype=torch.float16)
        # print(input_tensor["image"].meta)
        # print(input_tensor["image"].shape)
        # print(input_tensor["image"].device)

        output_array = run_inference(input_tensor, self.args, self.device, self.transform)
        # print(output_array.shape)
        #invert to match original
        output_array = np.transpose(output_array, (2, 1, 0))

        # print(f"Output shape: {output_array.shape}")
        # print(f"Output details: {output_array.max()}, {output_array.min()}")
        # print(f"Output details: {output_array.dtype}")

        output_image = sitk.GetImageFromArray(output_array)
        output_image.CopyInformation(input_image)
        
        return output_image


if __name__ == "__main__":
    ToothFairy3_OralPharyngealSegmentation().process()
