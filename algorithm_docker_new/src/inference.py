import torch
import gc
import time
import torch.nn as nn
import numpy as np
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.data.meta_tensor import MetaTensor

from .deep_watershed import deep_watershed_with_voting_optimized
from .inference_utils import merge_pulp_into_teeth_torch, remap_labels_torch, pred_to_challange_map
from .model import DWNet

def run_inference(input_tensor, args, device, transform) -> np.ndarray:

    model = DWNet(spatial_dims=3, in_channels=1, out_channels=args.out_channels, act=args.activation, norm=args.norm,
                bias=False, backbone_name=args.backbone_name, configuration=args.configuration)
    model.load_state_dict(torch.load('checkpoints/model_epoch_140.pth',
                                    map_location=device, weights_only=True)['model_state_dict'], strict=True)
    model = model.to(device)
    model.eval()

    with torch.no_grad(), torch.amp.autocast(enabled=True, dtype=torch.float16, device_type=device.type):
        print("Running inference on input tensor...")
        output = sliding_window_inference(input_tensor["image"], roi_size=args.patch_size, sw_batch_size=1, predictor=model, 
                                          overlap=0.5, sw_device=device, device='cpu', mode='gaussian', sigma_scale=0.125,
                                          padding_mode='constant', cval=0, progress=False)
        #delete model to free memory
        model.cpu()
        torch.cuda.empty_cache()

        (seg_multiclass, dist, pulp) = output
        # move to device
        seg_multiclass = seg_multiclass.to(device)
        dist = dist.to(device)
        pulp = pulp.to(device)  
        #get predictions
        multiclass_segmentation = seg_multiclass.argmax(dim=1, keepdim=True) # B,C, H,W,D
        dist_pred = nn.Threshold(1e-3, 0)(torch.sigmoid(dist))
        pulp_segmentation = (torch.sigmoid(pulp) > 0.5).float()
        binary_mask = torch.where(multiclass_segmentation >= 1, 1, 0)
        markers = torch.where(dist_pred > 0.5, 1, 0)
        #post_process
        pred_multiclass = deep_watershed_with_voting_optimized(dist_pred.squeeze().cpu().numpy(), 
                                                                multiclass_segmentation.squeeze().cpu().numpy(), 
                                                                binary_mask.squeeze().cpu().numpy(),
                                                                markers.squeeze().cpu().numpy())
        pred_multiclass = torch.from_numpy(pred_multiclass).long()
        pred_multiclass = pred_multiclass.to(device)  # H,W,D
        # print(pulp_segmentation.dtype, pulp_segmentation.shape, pulp_segmentation.device)
        # print(multiclass_segmentation.dtype, multiclass_segmentation.shape, multiclass_segmentation.device)
        # print(dist_pred.dtype, dist_pred.shape, dist_pred.device)

        # print(f"applied transforms - decollate: {decollate_batch(input_tensor)[0]['image'].applied_operations}")
        # print(f"applied transforms: {input_tensor['image'].applied_operations}")
        
        pred_with_pulp = merge_pulp_into_teeth_torch(pred_multiclass.squeeze(), pulp_segmentation.squeeze(), pulp_class=50) # H,W,D
        remapped = remap_labels_torch(pred_with_pulp, pred_to_challange_map)
        
        prediction = transform.post_inference_transform({"pred": MetaTensor(remapped.unsqueeze(0)), "image": input_tensor["image"][0]})["pred"] # B,H,W,D
        print(prediction.unique())
        
        return prediction.squeeze().cpu().numpy().astype(np.uint8) # H,W,D
        
        # pred_final = merge_pulp_into_teeth_torch(multiclass_segmentation, pulp_segmentation, pulp_class=111)
        # input_tensor['pred'] = multiclass_segmentation
        # output_array = [transform.post_inference_transform(i) for i in decollate_batch(input_tensor)] #transform cannot operate on a batch, we need C,H,W,D
    # return output_array[0].numpy().astype(np.uint8)  # Convert to numpy array and return
        
if __name__ ==  "__main__":
    start_time_epoch = time.time()
    run_inference()
    inference_time=time.time() - start_time_epoch
    print(f"Inference took: {inference_time:.2f}s.")