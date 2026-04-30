import torch
import gc
import time
import torch.nn as nn
import numpy as np
# from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.data.meta_tensor import MetaTensor
import torch.nn.functional as F
from copy import copy
from typing import Tuple, Optional
from monai.inferers.utils import compute_importance_map
from monai.transforms import SpatialCropD
from .sliding_window import _get_scan_interval, dense_patch_slices
from .sliding_window import sliding_window_inference
from .deep_watershed import deep_watershed_with_voting_optimized
from .inference_utils import merge_pulp_into_teeth_torch, remap_labels_torch, pred_to_challange_map
from .model import DWNet

import torch
import numpy as np

dtype_sizes = {
    torch.float16: 2,
    torch.float32: 4,
    torch.float64: 8,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 4,
    torch.int64: 8,
}

def load_model(args, device):
    model = DWNet(spatial_dims=3, in_channels=1, out_channels=args.out_channels, act=args.activation, norm=args.norm,
                bias=False, backbone_name=args.backbone_name, configuration=args.configuration)
    _, _ = model.load_state_dict(torch.load('checkpoints/model_epoch_380.pth',
                                    map_location=device, weights_only=True)['model_state_dict'], strict=False) #dir decoder weights are dropped
    model = model.to(device)
    model.eval()
    return model

def inference_wrapper(args, inference_fn, model, device, input_tensor, transform, patch_size, 
                      overlap, num_classes, prob_dtype=np.float16, mem_budget_bytes=12*1024**3):
    """
    Memory-aware wrapper for inference.
    Performs hard z-split if expected prob-map exceeds mem_budget.
    
    Args:
        input_tensor: torch.Tensor of shape (1, C, H, W, D)
        patch_size: tuple (Px, Py, Pz)
        overlap: float or tuple (for sliding window)
        num_classes: number of segmentation classes
        prob_dtype: float16 by default
        mem_budget_bytes: maximum memory for prob-map
    Returns:
        labels: np.uint8 array of shape (H, W, D)
    """
    _, _, H, W, D = input_tensor["image"].shape
    # estimate number of elements for full prob-map
    numel = H * W * D * num_classes
    bytes_needed = numel * torch.finfo(prob_dtype).bits // 8
    
    # if fits in memory, single pass
    if bytes_needed <= mem_budget_bytes:
        print("single step forward")
        labels = inference_fn(args, model, input_tensor, device) # inference_step(args, model, data_sample, transform, device)
        prediction = transform.post_inference_transform({"pred": MetaTensor(labels).unsqueeze(0), "image": input_tensor["image"][0]})["pred"] # B,H,W,D
        return prediction.squeeze().cpu().numpy().astype(np.uint8) # H,W,D
    
    print("multistep forward")
    # else: need hard z-split
    # compute safe max z-slab that fits in memory
    # print("memory too low, using slab inference")
    # max_voxels = mem_budget_bytes // (num_classes * dtype_sizes[prob_dtype])
    # z_slab_max = max_voxels // (H * W)
    # # optionally align to multiples of patch depth
    # pz = patch_size[2]
    # # z_slab_max = max(pz, (z_slab_max // pz) * pz)
    
    # # prepare z-slab ranges
    # # z_starts = list(range(0, D, z_slab_max))
    # # z_ranges = [(z0, min(z0 + z_slab_max, D)) for z0 in z_starts]
    
    # slab = min(z_slab_max, pz) if z_slab_max < 2*pz else z_slab_max  # keep it a multiple of Pz
    # If you want strictly fixed Pz-depth slabs, set: slab = Pz

    slab = patch_size[2] # step is always patch size z axis, cannot be less 
    plan = plan_z_slabs(D, slab)
    
    # allocate final label array
    labels = np.zeros((H, W, D), dtype=np.uint8)
    
    # process each slab
    for (zin0, zin1), (zout0, zout1) in plan:
        print("performing slab inference")
        # slab_tensor = {"image": input_tensor[:, :, :, :, z0:z1]}
        # cropper = SpatialCropD(keys="image", roi_start=(0, 0, zin0), roi_end=(H, W, zin1))
        slab_tensor = {"image": input_tensor["image"][:, :, :, :, zin0:zin1]}
        print(slab_tensor['image'].shape, slab_tensor['image'].device)
        # call main inference on slab
        # slab_output = inference_fn(args, model, slab_tensor, transform, device) 
        
        print("run step forward")
        slab_labels=inference_fn(args, model, slab_tensor,
                                 device).squeeze().to(torch.uint8).cpu().numpy()
        #assign slab fragment to final output
        off0 = zout0 - zin0
        off1 = zout1 - zin1  # negative or zero
        # if off1 != 0:
        #     slab_sub = slab_labels[:, :, off0:slab_labels.shape[2]+off1]
        # else:
        #     slab_sub = slab_labels[:, :, off0:]
        # labels[:, :, zout0:zout1] = slab_sub
        if off1 != 0:
            labels[:, :, zout0:zout1] = slab_labels[:, :, off0:slab_labels.shape[2] + off1]
        else:
            labels[:, :, zout0:zout1] = slab_labels[:, :, off0:]
        # Explicitly drop temporaries to keep RAM smooth
    
    # input_tensor["pred"] = MetaTensor(labels).unsqueeze(0).unsqueeze(0)
    # result = [transform.pred_transform(i) for i in decollate_batch(input_tensor)][0]['pred'] 
      
    prediction = transform.post_inference_transform({"pred": MetaTensor(labels).unsqueeze(0), "image": input_tensor["image"][0]})["pred"] # B,H,W,D
    return prediction.squeeze().to(torch.uint8).cpu().numpy() # H,W,D

def plan_z_slabs(D: int, slab: int):
    """
    Return a list of ((z_in0, z_in1), (z_out0, z_out1)) tuples.
    - z_in*:  slice passed to the model (always length == slab)
    - z_out*: sub-slice to write into the final volume to avoid overlaps
    """
    if D <= slab:
        # Single pass; no splitting. In+out are identical.
        return [((0, D), (0, D))]

    # Start positions spaced by `slab`, then force last start to be D - slab (end-aligned).
    starts = list(range(0, max(D - slab, 0) + 1, slab))
    last_start = D - slab
    if starts[-1] != last_start:
        starts.append(last_start)

    # Build (in, out) pairs: write [start_i : start_{i+1}], last writes [start_last : D]
    plan = []
    for i, s in enumerate(starts):
        z_in0, z_in1 = s, s + slab              # always slab deep
        z_out0 = s
        z_out1 = starts[i + 1] if i + 1 < len(starts) else D
        plan.append(((z_in0, z_in1), (z_out0, z_out1)))
    return plan

def roi_watershed_cpu(
    dist_pred: torch.Tensor,
    multiclass_segmentation: torch.Tensor,
    binary_mask: torch.Tensor,
    markers: torch.Tensor,
    threshold: float = 0.15,
    margin: int = 16,
    watershed_fn=None  # your deep_watershed_with_voting_optimized
) -> np.ndarray:
    """
    Apply watershed only on attention-based ROI for CPU tensors of shape [H,W,D].

    Args:
        dist_pred: attention map [H,W,D], CPU tensor
        multiclass_segmentation: multiclass prediction [H,W,D], CPU tensor
        binary_mask: binary mask [H,W,D], CPU tensor
        markers: watershed markers [H,W,D], CPU tensor
        threshold: attention threshold for ROI
        margin: voxels to pad around ROI
        watershed_fn: function performing watershed

    Returns:
        Full volume multiclass segmentation (np.ndarray)
    """

    # ROI mask from attention
    roi_mask = dist_pred > threshold

    if roi_mask.sum() == 0:
        # fallback: no high-attention region
        return watershed_fn(dist_pred.numpy(), 
                            multiclass_segmentation.numpy(), 
                            binary_mask.numpy(), 
                            markers.numpy())

    # Compute bounding box of ROI
    coords = torch.nonzero(roi_mask)
    min_coords = coords.min(dim=0)[0]
    max_coords = coords.max(dim=0)[0]

    # Add margin and clamp to image bounds
    min_coords = torch.clamp(min_coords - margin, min=0)
    max_coords = torch.clamp(max_coords + margin, max=torch.tensor(dist_pred.shape) - 1)

    # Create slices
    slices = tuple(slice(int(min_c), int(max_c)+1) for min_c, max_c in zip(min_coords, max_coords))

    # Extract ROI
    dist_roi = dist_pred[slices].numpy()
    multiclass_roi = multiclass_segmentation[slices].numpy()
    binary_roi = binary_mask[slices].numpy()
    markers_roi = markers[slices].numpy()

    # Apply watershed on ROI
    pred_multiclass_roi = watershed_fn(dist_roi, multiclass_roi, binary_roi, markers_roi)

    # Paste back into full volume
    pred_multiclass_full = np.zeros_like(multiclass_segmentation.numpy())
    pred_multiclass_full[slices] = pred_multiclass_roi

    return pred_multiclass_full

def memory_efficient_inference_with_overlap(
    args,
    model: nn.Module,
    input_tensor: torch.Tensor,
    patch_size: Tuple[int, int, int],
    device: Optional[torch.device] = None,
    overlap: float = 0.25,
    blend_mode: str = "gaussian", # "gaussian" or "constant"
    sigma_scale: float = 0.125,
    cast_dtype = torch.float16
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sliding-window inference with overlap + importance map (constant/gaussian).
    Supports mixed-precision inference (float16 accumulation).

    Args:
        model: segmentation model returning (seg_multiclass, dist, _, pulp).
        input_tensor: input image tensor [1, C, H, W, D].
        patch_size: ROI size (ph, pw, pd).
        stride: sliding step; defaults to (patch_size * (1 - overlap)).
        device: device for forward pass.
        overlap: fractional overlap between patches.
        mode: "gaussian" or "constant" weighting for blending.
        sigma_scale: used if mode="gaussian"; sigma = sigma_scale * dim_size.

    Returns:
        multiclass_output: [1,1,H,W,D] uint8 (argmax)
        dist_output:       [1,1,H,W,D] float16
        pulp_output:       [1,1,H,W,D] uint8
    """

    _, _, H, W, D = input_tensor.shape

    # Calculate scan intervals
    overlap = overlap if isinstance(overlap, (tuple, list)) else tuple([overlap]*len(patch_size))
    scan_interval = _get_scan_interval((H, W, D), patch_size, len(patch_size), overlap)
    
    # Enumerate all slices defining patches
    slices = dense_patch_slices((H, W, D), patch_size, scan_interval)

    # Preallocate accumulators
    sum_probs = torch.zeros((1, args.out_channels, H, W, D), dtype=cast_dtype, device="cpu")
    sum_weights = torch.zeros((1, 1, H, W, D), dtype=cast_dtype, device=device)
    dist_output = torch.zeros((1, 1, H, W, D), dtype=cast_dtype, device=device)
    pulp_output = torch.zeros((1, 1, H, W, D), dtype=cast_dtype,  device=device)

    # Create importance map (on device, but small, then copy to cpu when needed)
    importance_map = compute_importance_map(
        patch_size, mode=blend_mode, sigma_scale=sigma_scale, device=device, dtype=cast_dtype
    )  # [ph, pw, pd]
    importance_map = importance_map.unsqueeze(0).unsqueeze(0)  # [1,1,ph,pw,pd]

    with torch.no_grad():
        for s in slices:
            # Extract patch
            patch = input_tensor[:, :, s[0], s[1], s[2]]

            # Forward pass
            seg_multiclass, dist, pulp = model(patch)

            # Apply importance map
            w_patch = importance_map[:, :, :seg_multiclass.shape[2], :seg_multiclass.shape[3], :seg_multiclass.shape[4]]
            seg_multiclass = torch.softmax(seg_multiclass, dim=1).to(cast_dtype) * w_patch
            dist = torch.sigmoid(dist).to(cast_dtype) * w_patch
            pulp = torch.sigmoid(pulp).to(cast_dtype) * w_patch

            # Accumulate results
            sum_weights[:, :, s[0], s[1], s[2]] += w_patch # GPU stored
            sum_probs[:, :, s[0], s[1], s[2]] += seg_multiclass.cpu()
            dist_output[:, :, s[0], s[1], s[2]] += dist # GPU stored
            pulp_output[:, :, s[0], s[1], s[2]] += pulp # GPU stored

    # Final post-processing
    return (sum_probs / sum_weights.cpu().clamp(min=1e-6)).argmax(dim=1, keepdim=True).to(dtype=torch.uint8, device=device), \
            nn.Threshold(1e-3, 0)(dist_output / sum_weights.clamp(min=1e-6)).to(cast_dtype), \
            (pulp_output / sum_weights.clamp(min=1e-6) > 0.5).to(torch.uint8)

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
        # print("pred top inference")
        pred_top = sliding_window_inference(
            top_half, roi_size=roi_size, sw_batch_size=sw_batch_size,
            predictor=model, sw_device=device, device=device, **kwargs
        )
        pred_top = tuple(t.cpu() for t in pred_top)
        
        del top_half
        gc.collect()
        torch.cuda.empty_cache()
        
        # print("pred bottom inference")        
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
    # print("loading model...")
       
    model = DWNet(spatial_dims=3, in_channels=1, out_channels=args.out_channels, act=args.activation, norm=args.norm,
                bias=False, backbone_name=args.backbone_name, configuration=args.configuration)
    _, _ = model.load_state_dict(torch.load('checkpoints/model_epoch_380.pth',
                                    map_location=device, weights_only=True)['model_state_dict'], strict=False) #dir decoder weights are dropped
    model = model.to(device)
    model.eval()

    with torch.no_grad(), torch.amp.autocast(enabled=True, dtype=torch.float16, device_type=device.type):
        # num_voxels = input_tensor["image"].numel()
        # print(f"Running inference on input tensor with numel: {num_voxels}...")
       
        # if num_voxels > 40000000:
        #     output = split_infer_merge(input_tensor["image"], args.patch_size, model, device, split_overlap=4, mode='gaussian', overlap=0.1, sigma_scale=0.125,
        #                             padding_mode='constant', cval=0, progress=False)
        # else:
        #     output = sliding_window_inference(input_tensor["image"], roi_size=args.patch_size, sw_batch_size=1, predictor=model, 
        #                                     overlap=0.1, sw_device=device, device=device, mode='gaussian', sigma_scale=0.125,
        #                                     padding_mode='constant', cval=0, progress=False)
        #     output = tuple(t.cpu() for t in output)
        
        # output = memory_efficient_inference(model, input_tensor["image"], args.patch_size, device=device)
        # print("run sliding inference")
        output = memory_efficient_inference_with_overlap(args, model, input_tensor['image'], args.patch_size, device=device, overlap=0.5, 
                                                         blend_mode="gaussian", sigma_scale=0.125, cast_dtype=torch.float16)
        
        #unpack output
        #delete model to free memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # Unpack output and move to GPU
                
        # print("move to GPU")
        (multiclass_segmentation, dist_pred, pulp_segmentation) = output
        # print(f"seg device: {multiclass_segmentation.device}")
        # multiclass_segmentation = multiclass_segmentation.to(device)
        # pulp_segmentation = pulp_segmentation.to(device)
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
            
    # print("move to CPU")
    dist_pred_cpu = dist_pred.squeeze().cpu()
    multiclass_segmentation_cpu = multiclass_segmentation.squeeze().cpu()
    binary_mask_cpu =  binary_mask.squeeze().cpu()
    markers_cpu = markers.squeeze().cpu()
    
    # Free remaining GPU tensors
    del multiclass_segmentation, binary_mask, dist_pred, markers
    torch.cuda.empty_cache()
    
    # ---- CPU-based watershed (memory-intensive) ----
    # print("run Watershed")
    # FULL volume watershed
    pred_multiclass = deep_watershed_with_voting_optimized(dist_pred_cpu.numpy(), 
                                                           multiclass_segmentation_cpu.numpy(), 
                                                           binary_mask_cpu.numpy(),
                                                           markers_cpu.numpy())
    # attention based ROI watershed
    # USE AS LAST OPTION - UNSAFE
    # pred_multiclass = roi_watershed_cpu(dist_pred_cpu, multiclass_segmentation_cpu,
    #                                   binary_mask_cpu, markers_cpu, threshold=0.2, watershed_fn=deep_watershed_with_voting_optimized)
    # Free CPU copies used for watershed
    del dist_pred_cpu, multiclass_segmentation_cpu, binary_mask_cpu, markers_cpu
    gc.collect()
    
    with torch.no_grad():
        # ---- Move watershed result back to GPU ----
        # print("merge segmentations")
        pred_multiclass_gpu = torch.from_numpy(pred_multiclass).long().to(device)
        # pulp_segmentation = pulp_segmentation.to(device)
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
        
def inference_step(args, model, data_sample, device):
    with torch.no_grad(), torch.amp.autocast(enabled=True, dtype=torch.float16, device_type=device.type):
        
        output = memory_efficient_inference_with_overlap(args, model, data_sample['image'], args.patch_size, device=device, overlap=0.5, 
                                                        blend_mode="gaussian", sigma_scale=0.125, cast_dtype=torch.float16)
        
        #unpack output
        print("postprocessing")
        (multiclass_segmentation, dist_pred, pulp_segmentation) = output # all on GPU
        binary_mask = torch.where(multiclass_segmentation >= 1, 1, 0).to(torch.uint8)
        markers = torch.where(dist_pred > 0.5, 1, 0).to(torch.uint8)
        #CPU
        dist_pred_cpu = dist_pred.squeeze().cpu()
        multiclass_segmentation_cpu = multiclass_segmentation.squeeze().cpu()
        binary_mask_cpu =  binary_mask.squeeze().cpu()
        markers_cpu = markers.squeeze().cpu()
        
        del multiclass_segmentation, binary_mask, dist_pred, markers
        torch.cuda.empty_cache()
    
        #post_process
        pred_multiclass = deep_watershed_with_voting_optimized(dist_pred_cpu.numpy(), 
                                                                multiclass_segmentation_cpu.numpy(), 
                                                                binary_mask_cpu.numpy(), 
                                                                markers_cpu.numpy())
        
        
        del dist_pred_cpu, multiclass_segmentation_cpu, binary_mask_cpu, markers_cpu
        gc.collect()
        # pred_multiclass = deep_watershed_with_voting_optimized(to_numpy(dist_pred_cpu.numpy()), 
        #                                                        to_numpy(multiclass_segmentation_cpu.numpy()), 
        #                                                        to_numpy(binary_mask_cpu.numpy()), 
        #                                                        to_numpy(markers_cpu.numpy()))
        pred_multiclass_gpu = torch.from_numpy(pred_multiclass).to(device).long() # H,W,D
        del pred_multiclass
        gc.collect()   
        # ---- Merge pulp and remap labels ----``
        pred_with_pulp = merge_pulp_into_teeth_torch(pred_multiclass_gpu.squeeze(), pulp_segmentation.squeeze(), pulp_class=50).to(torch.int32)  # H,W,D
        remapped = remap_labels_torch(pred_with_pulp, pred_to_challange_map)
        
        del pred_with_pulp, pred_multiclass_gpu, pulp_segmentation
        torch.cuda.empty_cache()   
        
        return remapped
            
# if __name__ ==  "__main__":
#     start_time_epoch = time.time()
#     run_inference()
#     inference_time=time.time() - start_time_epoch
#     print(f"Inference took: {inference_time:.2f}s.")