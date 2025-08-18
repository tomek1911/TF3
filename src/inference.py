import torch
import time
import argparse
import yaml
import torch.nn as nn
import numpy as np
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, decollate_batch
from monai.data.dataset import Dataset

from deep_watershed import deep_watershed_with_voting
from inference_utils import merge_pulp_into_teeth
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
    input_image = {"image" : "data/imagesTr/ToothFairy3F_010_0000.nii.gz"}
    
    dataset = Dataset(data=[input_image], transform=transform.inference_preprocessing)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
    data_sample = next(iter(dataloader))
    
    # data_sample = transform.inference_preprocessing(input_image)
    model = DWNet(spatial_dims=3, in_channels=1, out_channels=args.out_channels, act=args.activation, norm=args.norm,
                bias=False, backbone_name=args.backbone_name, configuration=args.configuration)
    model.load_state_dict(torch.load('checkpoints/checkpoints/governing_raspberry_2678/model_epoch_100.pth',
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
        # binary_mask = torch.where(multiclass_segmentation >= 1, 1, 0)
        # markers = torch.where(dist_pred > 0.5, 1, 0)
        #post_process
        # pred_multiclass = deep_watershed_with_voting(to_numpy(dist_pred.squeeze()), 
        #                                              to_numpy(multiclass_segmentation.squeeze()), 
        #                                              to_numpy(binary_mask.squeeze()), 
        #                                              to_numpy(markers.squeeze()))
        # pred_final = merge_pulp_into_teeth(pred_multiclass, pulp_segmentation, pulp_class=111)
        
        data_sample['pulp'] = pulp_segmentation
        data_sample['dist'] = dist_pred
        data_sample['mlt'] = multiclass_segmentation
        inverted_prediction_mlt = [transform.post_inference_transform(i) for i in decollate_batch(data_sample)] 
        # inverted_prediction = transform.post_inference_transform(data_sample) 
        transform.save_inference_output(inverted_prediction_mlt[0])
        
if __name__ ==  "__main__":
    start_time_epoch = time.time()
    main()
    inference_time=time.time() - start_time_epoch
    print(f"Inference took: {inference_time:.2f}s.")