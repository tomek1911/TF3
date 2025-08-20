import torch
import gc
import time
import torch.nn as nn
import numpy as np
# from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.data.meta_tensor import MetaTensor
import torch.nn.functional as F
from typing import Tuple

from .sliding_window import sliding_window_inference
from .deep_watershed import deep_watershed_with_voting_optimized
from .inference_utils import merge_pulp_into_teeth_torch, remap_labels_torch, pred_to_challange_map
from .model import DWNet

def memory_efficient_inference(
    model: nn.Module,
    input_tensor: torch.Tensor,
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int] = None,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Memory-efficient inference for multi-output segmentation model.

    Args:
        model: the segmentation model
        input_tensor: [1, C, H, W, D] input
        patch_size: size of patches (h, w, d)
        stride: patch stride (h, w, d). Defaults to patch_size (non-overlapping)
        device: device for computation (default: same as input)

    Returns:
        multiclass_segmentation [1, 1, H, W, D] (uint8)
        distance map [1, 1, H, W, D] (float32)
        pulp segmentation [1, 1, H, W, D] (uint8)
    """
    device = device or input_tensor.device
    input_tensor = input_tensor.to(device)
    model = model.to(device).eval()

    _, _, H, W, D = input_tensor.shape
    ph, pw, pd = patch_size
    stride = stride or patch_size

    #initialise outputs in RAM
    multiclass_output = torch.zeros((1, 1, H, W, D), dtype=torch.uint8, device="cpu")
    dist_output = torch.zeros((1, 1, H, W, D), dtype=torch.float32, device="cpu")
    pulp_output = torch.zeros((1, 1, H, W, D), dtype=torch.uint8, device="cpu")

    for h0 in range(0, H, stride[0]):
        h1 = min(h0 + ph, H)
        for w0 in range(0, W, stride[1]):
            w1 = min(w0 + pw, W)
            for d0 in range(0, D, stride[2]):
                d1 = min(d0 + pd, D)

                # Extract patch within bounds
                patch = input_tensor[
                    :,
                    :,
                    h0:h1,
                    w0:w1,
                    d0:d1,
                ]

                # compute padding to reach patch_size
                pad_h = ph - (h1 - h0)
                pad_w = pw - (w1 - w0)
                pad_d = pd - (d1 - d0)

                if pad_h > 0 or pad_w > 0 or pad_d > 0:
                    patch = F.pad(
                        patch,
                        pad=(0, pad_d, 0, pad_w, 0, pad_h),  # (D_before,D_after, W_before,W_after, H_before,H_after)
                        mode="constant",
                        value=0
                    )

                # Forward pass
                with torch.no_grad():
                    seg_multiclass, dist, pulp = model(patch) # REMOVE _ for challange

                    # Post-processing
                    seg_patch = seg_multiclass.argmax(dim=1, keepdim=True).to(torch.uint8)
                    dist_patch = nn.Threshold(1e-3, 0)(torch.sigmoid(dist)).to(torch.float32)
                    pulp_patch = (torch.sigmoid(pulp) > 0.5).to(torch.uint8)
                    
                    # Crop padded areas to match valid region
                    h_end = h1 - h0
                    w_end = w1 - w0
                    d_end = d1 - d0
                    
                    # Place in output - on CPU
                    multiclass_output[:, :, h0:h1, w0:w1, d0:d1] =  seg_patch[:, :, :h_end, :w_end, :d_end].cpu()
                    dist_output[:, :, h0:h1, w0:w1, d0:d1] = dist_patch[:, :, :h_end, :w_end, :d_end].cpu()
                    pulp_output[:, :, h0:h1, w0:w1, d0:d1] =  pulp_patch[:, :, :h_end, :w_end, :d_end].cpu()

    return multiclass_output, dist_output, pulp_output

def split_infer_merge(
    image: torch.Tensor,
    roi_size,
    model,
    device,
    split_overlap: int = 0,
    sw_batch_size: int = 1,
    **kwargs
) -> torch.Tensor:
    """
    Run sliding_window_inference on two halves of the image along the z-axis,
    then merge the results with optional overlap.

    Args:
        image (torch.Tensor): Input tensor of shape (B, C, H, W, D).
        roi_size (tuple): ROI size for sliding_window_inference.
        model (torch.nn.Module): The predictor model.
        device (torch.device): Device for inference.
        overlap (int): Number of slices to overlap between halves along z-axis.
                       If 0, halves are non-overlapping.
        sw_batch_size (int): Sliding window batch size.
        **kwargs: Additional arguments to sliding_window_inference.

    Returns:
        torch.Tensor: Merged output of same shape as input (B, C, H, W, D).
    """
    B, C, H, W, D = image.shape
    mid = D // 2

    # Compute split indices with overlap
    start_top, end_top = 0, mid + split_overlap
    start_bottom, end_bottom = mid - split_overlap, D

    # Extract halves
    top_half = image[..., start_top:end_top]
    bottom_half = image[..., start_bottom:end_bottom]

    # Run inference on each half
    with torch.no_grad(), torch.amp.autocast(enabled=True, dtype=torch.float16, device_type=device.type):
        print("pred top inference")
        pred_top = sliding_window_inference(
            top_half, roi_size=roi_size, sw_batch_size=sw_batch_size,
            predictor=model, sw_device=device, device=device, **kwargs
        )
        pred_top = tuple(t.cpu() for t in pred_top)
        
        del top_half
        gc.collect()
        torch.cuda.empty_cache()
        
        print("pred bottom inference")
        
        pred_bottom = sliding_window_inference(
            bottom_half, roi_size=roi_size, sw_batch_size=sw_batch_size,
            predictor=model, sw_device=device, device=device, **kwargs
        )
        pred_bottom = tuple(t.cpu() for t in pred_bottom)
        
        del bottom_half
        gc.collect()
        torch.cuda.empty_cache()
        
        merged = []
        for t_top, t_bottom in zip(pred_top, pred_bottom):
            B, C, H, W, _ = t_top.shape
            out = torch.empty(B, C, H, W, D, device=t_top.device, dtype=t_top.dtype)

            # Top half
            out[..., :end_top] = t_top
            del t_top
            gc.collect()
            torch.cuda.empty_cache()

            # Bottom half
            if split_overlap > 0:
                # Average overlap region
                out[..., mid - split_overlap:mid] = (out[..., mid - split_overlap:mid] + t_bottom[..., :split_overlap]) / 2.0
                out[..., mid:] = t_bottom[..., split_overlap:]
            else:
                out[..., mid:] = t_bottom

            del t_bottom
            gc.collect()
            torch.cuda.empty_cache()
            merged.append(out)

    return tuple(merged)

def run_inference(input_tensor, args, device, transform) -> np.ndarray:
    print("loading model...")
       
    model = DWNet(spatial_dims=3, in_channels=1, out_channels=args.out_channels, act=args.activation, norm=args.norm,
                bias=False, backbone_name=args.backbone_name, configuration=args.configuration)
    missing_keys, unexpected_keys = model.load_state_dict(torch.load('checkpoints/model_epoch_260.pth',
                                    map_location=device, weights_only=True)['model_state_dict'], strict=False) #dir decoder weights are dropped
    model = model.to(device)
    model.eval()

    with torch.no_grad(), torch.amp.autocast(enabled=True, dtype=torch.float16, device_type=device.type):
        num_voxels = input_tensor["image"].numel()
        print(f"Running inference on input tensor with numel: {num_voxels}...")
       
        # if num_voxels > 40000000:
        #     output = split_infer_merge(input_tensor["image"], args.patch_size, model, device, split_overlap=4, mode='gaussian', overlap=0.1, sigma_scale=0.125,
        #                             padding_mode='constant', cval=0, progress=False)
        # else:
        #     output = sliding_window_inference(input_tensor["image"], roi_size=args.patch_size, sw_batch_size=1, predictor=model, 
        #                                     overlap=0.1, sw_device=device, device=device, mode='gaussian', sigma_scale=0.125,
        #                                     padding_mode='constant', cval=0, progress=False)
        #     output = tuple(t.cpu() for t in output)
        
        output = memory_efficient_inference(model, input_tensor["image"], args.patch_size, device=device)
        #delete model to free memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # Unpack output and move to GPU
                
        print("move to GPU")
        (multiclass_segmentation, dist_pred, pulp_segmentation) = output
        # seg_multiclass = seg_multiclass.to(device)
        # dist = dist.to(device)
        # pulp = pulp.to(device)  
        
        # Compute predictions on GPU
        # multiclass_segmentation = seg_multiclass.argmax(dim=1, keepdim=True).to(torch.uint8)  # B,C, H,W,D
        # dist_pred = nn.Threshold(1e-3, 0)(torch.sigmoid(dist)).to(torch.float32)
        # pulp_segmentation = (torch.sigmoid(pulp) > 0.5).float().to(torch.uint8) # values: 0-1
        binary_mask = torch.where(multiclass_segmentation >= 1, 1, 0).to(torch.uint8) # values: 0-1
        markers = torch.where(dist_pred > 0.5, 1, 0).to(torch.uint8) # values: 0-1
        
        # Free intermediate GPU tensors not needed for further GPU ops
        # del seg_multiclass, dist, pulp
        # torch.cuda.empty_cache()
        
    # ---- Move only necessary tensors to CPU for watershed ----
            
    print("move to CPU")
    dist_pred_cpu = dist_pred.squeeze().cpu()
    multiclass_segmentation_cpu = multiclass_segmentation.squeeze().cpu()
    binary_mask_cpu =  binary_mask.squeeze().cpu()
    markers_cpu = markers.squeeze().cpu()
    
    # Free remaining GPU tensors
    del multiclass_segmentation, binary_mask, dist_pred, markers
    torch.cuda.empty_cache()
    
    # ---- CPU-based watershed (memory-intensive) ----
    print("run Watershed")
    pred_multiclass = deep_watershed_with_voting_optimized(dist_pred_cpu.numpy(), 
                                                            multiclass_segmentation_cpu.numpy(), 
                                                            binary_mask_cpu.numpy(),
                                                            markers_cpu.numpy())
    # Free CPU copies used for watershed
    del dist_pred_cpu, multiclass_segmentation_cpu, binary_mask_cpu, markers_cpu
    gc.collect()
    
    with torch.no_grad():
        # ---- Move watershed result back to GPU ----
        print("merge segmentations")
        pred_multiclass_gpu = torch.from_numpy(pred_multiclass).long().to(device)
        pulp_segmentation = pulp_segmentation.to(device)
        del pred_multiclass
        gc.collect()   
    
        # ---- Merge pulp and remap labels ----
        pred_with_pulp = merge_pulp_into_teeth_torch(pred_multiclass_gpu.squeeze(), pulp_segmentation.squeeze(), pulp_class=50).to(torch.int32) 
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