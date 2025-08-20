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
    model.load_state_dict(torch.load('checkpoints/model_epoch_220.pth',
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

        # Unpack output and move to GPU
        (seg_multiclass, dist, pulp) = output
        seg_multiclass = seg_multiclass.to(device)
        dist = dist.to(device)
        pulp = pulp.to(device)  
        
        # Compute predictions on GPU
        multiclass_segmentation = seg_multiclass.argmax(dim=1, keepdim=True) # B,C, H,W,D
        binary_mask = torch.where(multiclass_segmentation >= 1, 1, 0)
        dist_pred = nn.Threshold(1e-3, 0)(torch.sigmoid(dist))
        pulp_segmentation = (torch.sigmoid(pulp) > 0.5).float()
        markers = torch.where(dist_pred > 0.5, 1, 0)
        
        # Free intermediate GPU tensors not needed for further GPU ops
        del seg_multiclass, dist, pulp
        torch.cuda.empty_cache()
        
    # ---- Move only necessary tensors to CPU for watershed ----
    dist_pred_cpu = dist_pred.squeeze().cpu()
    multiclass_segmentation_cpu = multiclass_segmentation.squeeze().cpu()
    binary_mask_cpu =  binary_mask.squeeze().cpu()
    markers_cpu = markers.squeeze().cpu()
    
    # Free remaining GPU tensors
    del multiclass_segmentation, binary_mask, dist_pred, markers
    torch.cuda.empty_cache()
    
    # ---- CPU-based watershed (memory-intensive) ----
    pred_multiclass = deep_watershed_with_voting_optimized(dist_pred_cpu.numpy(), 
                                                            multiclass_segmentation_cpu.numpy(), 
                                                            binary_mask_cpu.numpy(),
                                                            markers_cpu.numpy())
    # Free CPU copies used for watershed
    del dist_pred_cpu, multiclass_segmentation_cpu, binary_mask_cpu, markers_cpu
    gc.collect()
    
    with torch.no_grad():
        # ---- Move watershed result back to GPU ----
        pred_multiclass_gpu = torch.from_numpy(pred_multiclass).long().to(device)
        del pred_multiclass
        gc.collect()   
    
        # ---- Merge pulp and remap labels ----
        pred_with_pulp = merge_pulp_into_teeth_torch(pred_multiclass_gpu.squeeze(), pulp_segmentation.squeeze(), pulp_class=50)
        remapped = remap_labels_torch(pred_with_pulp, pred_to_challange_map)
        del pred_with_pulp, pred_multiclass_gpu, pulp_segmentation
        torch.cuda.empty_cache()   
        
        prediction = transform.post_inference_transform({"pred": MetaTensor(remapped.unsqueeze(0)), "image": input_tensor["image"][0]})["pred"] # B,H,W,D
        
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