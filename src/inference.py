import torch
import time
import argparse
import yaml
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, decollate_batch
from monai.data.dataset import Dataset
from monai.data.meta_tensor import MetaTensor

from deep_watershed import deep_watershed_with_voting, deep_watershed_with_voting_optimized
from inference_utils import merge_pulp_into_teeth, merge_pulp_into_teeth_torch, remap_labels_torch, pred_to_challange_map
from transforms import Transforms
from model import DWNet

def main():

    config_file = 'config.yaml'
    with open(config_file, 'r') as file:
        general_config = yaml.safe_load(file)
    args = argparse.Namespace(**general_config['args'])

    def get_default_device():
        """Set device for computation"""
        if torch.cuda.is_available():
            return torch.device('cuda:1')
        return torch.device('cpu')
    
    def to_numpy(t: torch.Tensor) -> np.ndarray:
        return t.cpu().numpy()

    device = get_default_device()
    transform = Transforms(args, device=device)
    input_image = {"image" : "data/imagesTr/ToothFairy3P_381_0000.nii.gz"}
    
    dataset = Dataset(data=[input_image], transform=transform.inference_preprocessing)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
    data_sample = next(iter(dataloader))
    
    # data_sample = transform.inference_preprocessing(input_image)
    model = DWNet(spatial_dims=3, in_channels=1, out_channels=args.out_channels, act=args.activation, norm=args.norm,
                bias=False, backbone_name=args.backbone_name, configuration=args.configuration)
    model.load_state_dict(torch.load('checkpoints/checkpoints/silent_pie_5061/model_epoch_140.pth',
                                    map_location=device, weights_only=True)['model_state_dict'], strict=True)
    model = model.to(device)
    model.eval()

    with torch.no_grad(), torch.amp.autocast(enabled=True, dtype=torch.float16, device_type=device.type):
        output = sliding_window_inference(data_sample["image"], roi_size=args.patch_size, sw_batch_size=1, predictor=model, 
                                          overlap=0.5, sw_device=device, device='cpu', mode='gaussian', sigma_scale=0.125,
                                          padding_mode='constant', cval=0, progress=False)
        #unpack output
        (seg_multiclass, dist, _, pulp) = output
        #get predictions
        multiclass_segmentation = seg_multiclass.argmax(dim=1, keepdim=True)
        pulp_segmentation = (torch.sigmoid(pulp) > 0.5).float()
        dist_pred = nn.Threshold(1e-3, 0)(torch.sigmoid(dist))
        binary_mask = torch.where(multiclass_segmentation >= 1, 1, 0)
        markers = torch.where(dist_pred > 0.5, 1, 0)
        #post_process
        # pred_multiclass = deep_watershed_with_voting(to_numpy(dist_pred.squeeze()), 
        #                                              to_numpy(multiclass_segmentation.squeeze()), 
        #                                              to_numpy(binary_mask.squeeze()), 
        #                                              to_numpy(markers.squeeze()))
        pred_multiclass = deep_watershed_with_voting_optimized(to_numpy(dist_pred.squeeze()), 
                                                               to_numpy(multiclass_segmentation.squeeze()), 
                                                               to_numpy(binary_mask.squeeze()), 
                                                               to_numpy(markers.squeeze()))
        pred_multiclass = torch.from_numpy(pred_multiclass).long() # H,W,D
        # pred_final = merge_pulp_into_teeth(pred_multiclass, pulp_segmentation, pulp_class=111)
        
        multiclass_segmentation = merge_pulp_into_teeth_torch(pred_multiclass.squeeze(), pulp_segmentation.squeeze(), pulp_class=50) # H,W,D
        remapped = remap_labels_torch(multiclass_segmentation, pred_to_challange_map)
        
        data_sample['pulp'] = pulp_segmentation
        data_sample['dist'] = dist_pred
        data_sample['mlt'] = MetaTensor(remapped.unsqueeze(0).unsqueeze(0)) # HERE IS CHANGED INPUT
        inverted_prediction_mlt = [transform.post_inference_transform_no_dir(i) for i in decollate_batch(data_sample)] 
        # inverted_prediction = transform.post_inference_transform(data_sample) 
        transform.save_inference_output(inverted_prediction_mlt[0])


def sitk_to_monai(image: sitk.Image):
    """
    Convert a SimpleITK image to a numpy array + MONAI-style meta dict.
    Handles channel dimension and affine info.
    """
    array = sitk.GetArrayFromImage(image)  # shape [D,H,W]
    array = np.expand_dims(array, axis=0)  # add channel dim [C,D,H,W]

    spacing = image.GetSpacing()  # x,y,z
    origin = image.GetOrigin()
    direction = np.array(image.GetDirection()).reshape(3,3)

    # Build 4x4 affine
    affine = np.eye(4)
    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3, 3] = origin

    meta = {
        "original_affine": affine,
        "affine": affine.copy(),
        "dim": array.shape,
    }

    return array, meta

import nibabel as nib
from nibabel.orientations import aff2axcodes

def inspect_nifti(path: str):
    # Load NIfTI
    img = nib.load(path)

    # Get affine matrix
    affine = img.affine
    print("Affine:\n", affine)

    # Derive orientation codes (e.g. RAS, LPS, RPI, etc.)
    axcodes = aff2axcodes(affine)
    orientation = "".join(axcodes)
    print("Detected orientation:", orientation)

    return affine, orientation

if __name__ ==  "__main__":
    # from monai.transforms import LoadImage
    # image = LoadImage(reader='NibabelReader')('data/imagesTr/ToothFairy3P_077_0000.nii.gz')
    # affine, orientation = inspect_nifti("data/imagesTr/ToothFairy3P_077_0000.nii.gz")

    start_time_epoch = time.time()
    main()
    inference_time=time.time() - start_time_epoch
    print(f"Inference took: {inference_time:.2f}s.")