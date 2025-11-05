import torch
import time
import argparse
import yaml
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np
import gc
import SimpleITK as sitk
import nibabel as nib
from nibabel.orientations import aff2axcodes

from contextlib import nullcontext
from copy import copy
from monai.data import DataLoader, decollate_batch
from monai.data.dataset import Dataset
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import compute_importance_map, BlendMode
from monai.inferers.utils import compute_importance_map

from deep_watershed import deep_watershed_with_voting, deep_watershed_with_voting_optimized
from inference_utils import merge_pulp_into_teeth_torch, remap_labels_torch, pred_to_challange_map, save_nifti
from transforms import Transforms, SaveMultipleKeysD
from sliding_window import sliding_window_inference, _get_scan_interval, dense_patch_slices
from model import DWNet
from monai.transforms import SpatialCropD

dtype_sizes = {
    torch.float16: 2,
    torch.float32: 4,
    torch.float64: 8,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 4,
    torch.int64: 8,
}

def free(tensor):
    if tensor is not None:
        tensor = None
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return tensor

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
        labels = inference_fn(args, model, input_tensor, transform, device) # inference_step(args, model, data_sample, transform, device)
        input_tensor["pred"] = MetaTensor(labels).unsqueeze(0).unsqueeze(0)
        result = [transform.pred_transform(i) for i in decollate_batch(input_tensor)][0]['pred'] 
        return result.squeeze().cpu().numpy().astype(np.uint8)
    
    # else: need hard z-split
    # compute safe max z-slab that fits in memory
    max_voxels = mem_budget_bytes // (num_classes * dtype_sizes[prob_dtype])
    z_slab_max = max_voxels // (H * W)
    pz = patch_size[2]
    # slab = min(z_slab_max, pz) if z_slab_max < 2*pz else z_slab_max
    slab = pz[2] #z step is always patch size, cannot be smaller

    plan = plan_z_slabs(D, slab)
    
    # allocate final label array
    labels = np.zeros((H, W, D), dtype=np.uint8)
    
    # process each slab
    for (zin0, zin1), (zout0, zout1) in plan:
        # slab_tensor = {"image": input_tensor[:, :, :, :, z0:z1]}
        cropper = SpatialCropD(keys="image", roi_start=(0, 0, zin0), roi_end=(H, W, zin1))
        #shallow copy
        slab_tensor = copy(input_tensor)
        slab_tensor["image"] = input_tensor["image"][0]
        slab_tensor = cropper(slab_tensor)
        slab_tensor["image"] = slab_tensor["image"].unsqueeze(0)
        # call main inference on slab
        # slab_output = inference_fn(args, model, slab_tensor, transform, device) 
        
        slab_labels=inference_fn(args, model, slab_tensor, transform, device).cpu().numpy().astype(np.uint8)
        #assign slab fragment to final output
        off0 = zout0 - zin0
        off1 = zout1 - zin1  # negative or zero
        if off1 != 0:
            slab_sub = slab_labels[:, :, off0:slab_labels.shape[2]+off1]
        else:
            slab_sub = slab_labels[:, :, off0:]
        labels[:, :, zout0:zout1] = slab_sub

        # Explicitly drop temporaries to keep RAM smooth
        del slab_sub, slab_labels, slab_tensor
    
    input_tensor["pred"] = MetaTensor(labels).unsqueeze(0).unsqueeze(0)
    result = [transform.pred_transform(i) for i in decollate_batch(input_tensor)][0]['pred'] 
    return result.squeeze().cpu().numpy().astype(np.uint8)

def memory_efficient_inference_with_overlap(
    args,
    model: nn.Module,
    input_tensor: torch.Tensor,
    patch_size: Tuple[int, int, int],
    device: Optional[torch.device] = None,
    overlap: float = 0.25,
    blend_mode: str = "gaussian", # "gaussian" or "constant"
    sigma_scale: float = 0.125,
    cast_dtype = torch.float16,
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
    return (sum_probs / sum_weights.cpu().clamp(min=1e-6)).argmax(dim=1).to(dtype=torch.uint8, device=device), \
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

def memory_efficient_inference_2(
    args,
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    patch_size: Tuple[int, int, int],
    device: Optional[torch.device] = None,
    overlap: float = 0.25,
    blend_mode: str = "gaussian",
    sigma_scale: float = 0.125,
    cast_dtype: torch.dtype = torch.float16,
    dist_map_cast_dtype = torch.float32,
    memory_efficient: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Sliding-window inference with overlap.
    Supports memory-efficient mode with vectorized class update.
    """
    device = device or input_tensor.device
    _, _, H, W, D = input_tensor.shape

    overlap = overlap if isinstance(overlap, (tuple, list)) else (overlap,) * len(patch_size)
    scan_interval = _get_scan_interval((H, W, D), patch_size, len(patch_size), overlap)
    slices = dense_patch_slices((H, W, D), patch_size, scan_interval)

    # Importance map
    importance_map = compute_importance_map(
        patch_size, mode=blend_mode, sigma_scale=sigma_scale, device=device, dtype=cast_dtype
    )[None, None] # [1,1,ph,pw,pd]

    if memory_efficient:
        # --- Vectorized memory-efficient mode ---
        best_class = torch.zeros((1, 1, H, W, D), dtype=torch.uint8, device="cpu", pin_memory=True)
        best_score = torch.zeros((1, 1, H, W, D), dtype=torch.float32, device="cpu", pin_memory=True)
        sum_weights = torch.zeros((1, 1, H, W, D), dtype=torch.float32, device="cpu", pin_memory=True)

        for s in slices:

            patch = input_tensor[:, :, s[0], s[1], s[2]].to(device, non_blocking=True)
            logits, dist, pulp = model(patch)

            probs_patch = torch.softmax(logits, dim=1).to(torch.float32)
            w_patch = importance_map[:, :, :probs_patch.shape[2],
                                        :probs_patch.shape[3],
                                        :probs_patch.shape[4]]
            # Move to CPU
            probs_patch = probs_patch.cpu()
            w_patch = w_patch.cpu()

            # Weighted probabilities
            weighted_probs = probs_patch * w_patch  # [1,C,ph,pw,pd]

            # Vectorized argmax along class dimension
            local_max_score, local_class = weighted_probs.max(dim=1, keepdim=True)  # [1,1,ph,pw,pd]

            # Global slice
            gs = (s[0], s[1], s[2])

            # Update sum_weights
            sum_weights[:, :, gs[0], gs[1], gs[2]] += w_patch

            # Update global best_score and best_class
            existing_best = best_score[:, :, gs[0], gs[1], gs[2]]
            mask = local_max_score > existing_best
            best_score[:, :, gs[0], gs[1], gs[2]][mask] = local_max_score[mask]
            best_class[:, :, gs[0], gs[1], gs[2]][mask] = local_class[mask]

        multiclass_output = best_class
        dist_output = None
        pulp_output = None

    else:
        #CPU accumulator
        multiclass_probs = torch.zeros((1, args.out_channels, H, W, D), dtype=cast_dtype, device="cpu")
        #GPU accumulator
        weights_accumulator = torch.zeros((1, 1, H, W, D), dtype=cast_dtype, device=device)
        dist_probs = torch.zeros((1, 1, H, W, D), dtype=dist_map_cast_dtype, device=device)
        pulp_probs = torch.zeros((1, 1, H, W, D), dtype=cast_dtype, device=device)
        
        for s in slices:
            patch = input_tensor[:, :, s[0], s[1], s[2]]
            logits, dist_patch, pulp_patch = model(patch)
            
            #clamp logits to avoid oversaturation in float16
            weighted_probs = torch.softmax(logits.clamp(-15,15), dim=1).to(cast_dtype) * importance_map              
            weighted_dist = torch.sigmoid(dist_patch.clamp(-8,8)).to(dist_map_cast_dtype) * importance_map
            weighted_pulp = torch.sigmoid(pulp_patch.clamp(-8,8)).to(cast_dtype) * importance_map
            
            weights_accumulator[:, :, s[0], s[1], s[2]] += importance_map
            dist_probs[:, :, s[0], s[1], s[2]] += weighted_dist
            pulp_probs[:, :, s[0], s[1], s[2]] += weighted_pulp
            torch.cuda.synchronize() # ensure GPU->CPU transfer is done
            multiclass_probs[:, :, s[0], s[1], s[2]] += weighted_probs.to('cpu', dtype=cast_dtype)

    return multiclass_probs, dist_probs, pulp_probs, weights_accumulator

def memory_efficient_inference_3(
    args,
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    patch_size: Tuple[int, int, int],
    device: Optional[torch.device] = None,
    overlap: float = 0.25,
    blend_mode: str = "gaussian",
    sigma_scale: float = 0.125,
    cast_dtype: torch.dtype = torch.float16,
    dist_map_cast_dtype=torch.float32,
    non_blocking: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sliding-window inference with overlap and memory-efficient CPU accumulation.
    Flushes non-overlapping slabs of predictions progressively to limit RAM usage.
    Compatible with MONAI `_get_scan_interval` and `dense_patch_slices`.
    """

    device = device or input_tensor.device
    _, _, H, W, D = input_tensor.shape

    overlap = overlap if isinstance(overlap, (tuple, list)) else (overlap,) * len(patch_size)
    scan_interval = _get_scan_interval((H, W, D), patch_size, len(patch_size), overlap)
    slices = dense_patch_slices((H, W, D), patch_size, scan_interval)
     # Sort slices by z_start
    slices.sort(key=lambda s: (s[2].start, s[1].start, s[0].start))

    # Gaussian importance map for weighting
    importance_map = compute_importance_map(
        patch_size, mode=blend_mode, sigma_scale=sigma_scale, device=device, dtype=cast_dtype
    )[None, None]  # shape (1,1,ph,pw,pd)

     # Final output segmentation (uint8)
    pred_seg = torch.zeros((1, 1, H, W, D), dtype=torch.uint8, device="cpu")
    
    # Accumulators (1-channel)
    weights_accumulator = torch.zeros((1, 1, H, W, D), dtype=cast_dtype, device=device)
    dist_probs = torch.zeros((1, 1, H, W, D), dtype=dist_map_cast_dtype, device=device)
    pulp_probs = torch.zeros((1, 1, H, W, D), dtype=cast_dtype, device=device)
    
    # Temporary slab for class probabilities
    slab_probs = torch.zeros((1, args.out_channels, H, W, patch_size[2]), dtype=cast_dtype, device="cpu")

    # Flush helpers
    nx = math.ceil((H - patch_size[0]) / scan_interval[0]) + 1
    ny = math.ceil((W - patch_size[1]) / scan_interval[1]) + 1
    patches_per_z = nx * ny
    last_flushed_z = 0

    with torch.inference_mode():
        for idx, s in enumerate(slices):
            z0, z1 = s[2].start, s[2].stop

            # Extract patch and run inference
            patch = input_tensor[:, :, s[0], s[1], s[2]].to(device, non_blocking=non_blocking)
            logits, dist_patch, pulp_patch = model(patch)

            # Compute weighted probabilities on GPU
            weighted_probs = torch.softmax(logits.clamp(-15, 15), dim=1).to(cast_dtype) * importance_map
            weighted_dist = torch.sigmoid(dist_patch.clamp(-8, 8)).to(dist_map_cast_dtype) * importance_map
            weighted_pulp = torch.sigmoid(pulp_patch.clamp(-8, 8)).to(cast_dtype) * importance_map

            torch.cuda.synchronize()
            weighted_probs_cpu = weighted_probs.to("cpu", non_blocking=non_blocking)
           
            slab_probs[:, :, s[0], s[1], :] += weighted_probs_cpu
            weights_accumulator[:, :, s[0], s[1], s[2]] += importance_map
            dist_probs[:, :, s[0], s[1], s[2]] += weighted_dist
            pulp_probs[:, :, s[0], s[1], s[2]] += weighted_pulp

            del patch, logits, weighted_probs, weighted_dist, weighted_pulp, weighted_probs_cpu
            torch.cuda.empty_cache()
            
            # Flush once we complete all XY patches for a z block
            if (idx + 1) % patches_per_z == 0:
                z_end = z1
                flush_end = max(last_flushed_z, min(D, z_end - int(patch_size[2] * overlap[2])))
                
                slab = slab_probs[:, :, :, :, last_flushed_z:flush_end]
                slab /= weights_accumulator[:, :, :, :, last_flushed_z:flush_end].cpu()
                pred_seg[:, :, :, :, last_flushed_z:flush_end] = slab.argmax(dim=1)
                last_flushed_z = flush_end
                #transfer slab overlap to the begging, zero previous probs 
                slab_probs[:, :, :, :, z0:(z_end-flush_end)] = slab_probs[:, :, :, :, flush_end:z_end]
                slab_probs[:, :, :, :, (z_end-flush_end):] = 0
                
         # Final flush (if needed)
        if last_flushed_z < D:
            slab = slab_probs[:, :, :, :, last_flushed_z:]
            weights = weights_accumulator[:, :, :, :, last_flushed_z:]
            slab /= torch.clamp_min(weights, 1e-6)
            slab_probs[:, :, :, :, last_flushed_z:] = slab
            

    return pred_seg, dist_probs, pulp_probs, weights_accumulator

import torch
import numpy as np
import math
from typing import Tuple, Optional


def memory_efficient_inference_final(
    args,
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    patch_size: Tuple[int, int, int],
    device: Optional[torch.device] = None,
    device_aux: str = "gpu",
    overlap: float = 0.25,
    blend_mode: str = "gaussian",
    sigma_scale: float = 0.125,
    cast_dtype: torch.dtype = torch.float16,
    dist_map_cast_dtype=torch.float32,
    normalize_to_fp32: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Memory-efficient sliding-window inference with correct slab carry-over.

    - Uses a single slab buffer `slab_probs` (CPU) of depth `slab_depth = pz + overlap_z`.
    - Writes incoming patch contributions into local slab indices.
    - After processing all XY patches for a z_start, flush the front (non-overlapping) part,
      convert to uint8 labels, write to `pred_seg`, then shift the retained overlap to the
      beginning of the slab buffer. No full multiclass volume is kept.
    - Keeps full-volume weights/dist/pulp for now.
    """

    device = device or input_tensor.device
    if device_aux == "gpu":
        device_aux = device 
    else:
        device_aux = torch.device("cpu")
    
    _, _, H, W, D = input_tensor.shape  # input layout: (B=1, C_in, H, W, D)

    # normalize overlap param into tuple
    overlap = overlap if isinstance(overlap, (tuple, list)) else (overlap,) * len(patch_size)

    # compute scan intervals and dense slices using your existing utils
    scan_interval = _get_scan_interval((H, W, D), patch_size, len(patch_size), overlap)
    slices = dense_patch_slices((H, W, D), patch_size, scan_interval)

    # build list of unique z starts (in the same way dense_patch_slices generates them)
    # helper to compute starts like dense_patch_slices does
    def _starts(length, patch, stride):
        if length <= patch:
            return [0]
        starts = list(range(0, length - patch + 1, stride))
        if starts[-1] + patch < length:
            starts.append(length - patch)
        return starts

    z_starts = _starts(D, patch_size[2], scan_interval[2])
    # y_starts = _starts(W, patch_size[1], scan_interval[1])
    # x_starts = _starts(H, patch_size[0], scan_interval[0])

    # Group slices by z_start for robust processing (don't rely on dense_patch_slices ordering)
    # Convert slices list to mapping: z_start -> list of slice-tuples
    slices_by_z = {}
    for s in slices:
        z0 = s[2].start
        slices_by_z.setdefault(z0, []).append(s)

    # Sort z_starts ascending (should already be), and ensure mapping exists
    z_starts = sorted(z_starts)
    # sanity check: every z_start should be present as key in slices_by_z
    # If not present, add empty list to preserve ordering
    for z0 in z_starts:
        slices_by_z.setdefault(z0, [])

    # Derived params
    pz = patch_size[2]
    stride_z = scan_interval[2]  # step between z starts as computed by _get_scan_interval
    overlap_z = max(0, pz - stride_z)  # may be 0 or positive
    slab_depth = pz + overlap_z  # buffer depth to hold current pz + the overlap carried to next slab

    # Importance map expected shape: (1,1,ph,pw,pz) matching your code
    importance_map = compute_importance_map(
        patch_size, mode=blend_mode, sigma_scale=sigma_scale, device=device, dtype=cast_dtype
    )[None, None]  # (1,1,ph,pw,pz)

    # Output segmentation (uint8) kept in CPU
    pred_seg = torch.zeros((1, H, W, D), dtype=torch.uint8, device="cpu")

    # Keep full-volume 1-channel accumulators as you requested (CPU)
    weights_accumulator = torch.zeros((1, 1, H, W, D), dtype=cast_dtype, device=device_aux)
    dist_probs = torch.zeros((1, 1, H, W, D), dtype=dist_map_cast_dtype, device=device_aux)
    pulp_probs = torch.zeros((1, 1, H, W, D), dtype=cast_dtype, device=device_aux)

    # The slab buffer lives on CPU and maps to global Z region [slab_start, slab_start + slab_depth)
    slab_probs = torch.zeros((1, args.out_channels, H, W, slab_depth), dtype=cast_dtype, device='cpu')

    # Initialize slab_start at the first z_start
    slab_start = z_starts[0]
    # We will ensure slab_start always equals the global Z coordinate of slab_probs[:, :,:,:,0]
    # valid region within slab (how many slices currently filled) is tracked implicitly by slab contents,
    # but mapping uses absolute coordinates.

    # Process each z_start in order
    n_z = len(z_starts)
    with torch.inference_mode():
        for z_idx, z0 in enumerate(z_starts):
            # Process all XY patches that start at z = z0
            for s in slices_by_z[z0]:
                # slice tuples s = (slice_h, slice_w, slice_z)
                # patch extraction consistent with your code: input_tensor[:, :, s[0], s[1], s[2]]
                patch = input_tensor[:, :, s[0], s[1], s[2]].to(device, non_blocking=False)  # blocking to be safe
                logits, dist_patch, pulp_patch = model(patch)

                # compute weighted probabilities on GPU
                weighted_probs = torch.softmax(logits.clamp(-15, 15), dim=1).to(cast_dtype)  # (1,C,ph,pw,pz)
                weighted_probs = weighted_probs * importance_map  # broadcasting ok

                weighted_dist = torch.sigmoid(dist_patch.clamp(-8, 8)).to(dist_map_cast_dtype) * importance_map
                weighted_pulp = torch.sigmoid(pulp_patch.clamp(-8, 8)).to(cast_dtype) * importance_map

                # Move to CPU (blocking copy)
                weighted_probs = weighted_probs.to("cpu")           # (1,C,ph,pw,pz)
                if device_aux != device:
                    weighted_dist = weighted_dist.to(device_aux)            # (1,1,ph,pw,pz)
                    weighted_pulp = weighted_pulp.to(device_aux)         # (1,1,ph,pw,pz)
                    importance_map = importance_map.to(device_aux)          # (1,1,ph,pw,pz)

                # Map patch Z-range into local slab coordinates
                z_patch0 = s[2].start
                z_patch1 = s[2].stop
                local_z0 = z_patch0 - slab_start
                local_z1 = z_patch1 - slab_start
                if not (0 <= local_z0 <= slab_depth and 0 <= local_z1 <= slab_depth):
                    # If patch extends beyond current slab buffer, that's unexpected:
                    # it means slab_start is not aligned with z0. This can happen only if
                    # slab_start was advanced beyond z0 incorrectly. Raise explicit error.
                    raise RuntimeError(
                        f"Patch z-range [{z_patch0},{z_patch1}) not mappable into slab "
                        f"[{slab_start},{slab_start+slab_depth}). "
                        "This indicates logic bug in slab_start tracking."
                    )

                # Accumulate into slab buffer and full-volume 1-channel maps
                slab_probs[:, :, s[0], s[1], local_z0:local_z1] += weighted_probs
                weights_accumulator[:, :, s[0], s[1], s[2]] += importance_map
                dist_probs[:, :, s[0], s[1], s[2]] += weighted_dist
                pulp_probs[:, :, s[0], s[1], s[2]] += weighted_pulp

                # free GPU/large tensors
                del patch, logits, weighted_probs, weighted_dist, weighted_pulp
                torch.cuda.empty_cache()

            # Finished all XY patches for current z0.
            # Compute how many leading slices are now safe to flush.
            if z_idx < (n_z - 1):
                # Normal case: next z_start exists; flush up to z0 + stride_z (no further patches will touch below that)
                next_z0 = z_starts[z_idx + 1]
                # stride is next_z0 - z0, and flush_end is z0 + stride (equivalently next_z0)
                flush_end = z0 + (next_z0 - z0)  # equals next_z0
            else:
                # Last z block: flush everything remaining in slab up to image end
                flush_end = D

            # We need to clamp flush_end to slab coverage: valid flush range is [slab_start, slab_start + slab_depth)
            # flush_len is the number of slices from slab_start we can flush now
            flush_end_clamped = min(flush_end, slab_start + slab_depth, D)
            flush_len = flush_end_clamped - slab_start
            if flush_len > 0:
                # Normalize and argmax on the flushable portion
                probs_slice = slab_probs[:, :, :, :, :flush_len]  # (1,C,H,W,flush_len)
                weights_slice = weights_accumulator[:, :, :, :, slab_start: slab_start + flush_len].to('cpu')  # (1,1,H,W,flush_len)
                # Convert to float32 for stable division if cast_dtype is float16
                if normalize_to_fp32:
                    probs_slice = (probs_slice.float() / weights_slice.float()).to(cast_dtype)
                else:
                    probs_slice /= weights_slice
                # argmax -> (1,H,W,flush_len)
                pred_slice = torch.argmax(probs_slice, dim=1).to(torch.uint8)  # (1,H,W,flush_len)
                # write to final segmentation map
                pred_seg[:, :, :, slab_start: slab_start + flush_len] = pred_slice

                # If this is the last flush to end (flush_end == D), we're done; no need to shift buffer
                if flush_end_clamped == D:
                    # Done: break out early if desired (or continue loop; no more data will be added)
                    # But ensure we do not attempt to shift beyond image end.
                    # Clear slab and break
                    # (No need to zero out the slab; function will return)
                    slab_probs.zero_()  # optional free
                    slab_start += flush_len  # equals D
                    break

                # Otherwise, we must shift the remaining overlap [flush_len : slab_depth) to the front
                remaining = slab_depth - flush_len
                if remaining > 0:
                    # Move remaining slice block to the front
                    slab_probs[:, :, :, :, :remaining] = slab_probs[:, :, :, :, flush_len: flush_len + remaining]
                    # Zero the tail region to prepare for new incoming contributions
                    slab_probs[:, :, :, :, remaining:] = 0
                else:
                    slab_probs.zero_()

                # Advance slab_start by flush_len
                slab_start += flush_len

                # Note: weights_accumulator is global full-volume and we intentionally do NOT zero flushed weights here,
                # because you requested to keep weights/dist/pulp for later. If you want to free them, you can zero
                # weights_accumulator[..., slab_start - flush_len : slab_start] here to reclaim memory.

        # End z loop. If slab_start < D (e.g., due to early break not occuring), do final flush
        if slab_start < D:
            # Remaining length
            remaining_len = min(slab_depth, D - slab_start)
            if remaining_len > 0:
                probs_slice = slab_probs[:, :, :, :, :remaining_len].to(torch.float32)
                denom = weights_accumulator[:, :, :, :, slab_start: slab_start + remaining_len].to(torch.float32)
                probs_slice = probs_slice / torch.clamp_min(denom, 1e-6)
                pred_slice = torch.argmax(probs_slice, dim=1).to(torch.uint8)
                pred_seg[:, :, :, slab_start: slab_start + remaining_len] = pred_slice
            # any further tail beyond slab buffer should already be handled because last z_start aligns so last patch reaches D

    # Return segmentation (uint8) and full-volume aux outputs
    return pred_seg, dist_probs, pulp_probs, weights_accumulator

def split_infer_merge(
    image: torch.Tensor,
    roi_size,
    model,
    device,
    overlap: int = 0,
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
    start_top, end_top = 0, mid + overlap
    start_bottom, end_bottom = mid - overlap, D

    # Extract halves
    top_half = image[..., start_top:end_top]
    bottom_half = image[..., start_bottom:end_bottom]

    # Run inference on each half
    with torch.no_grad(), torch.amp.autocast(enabled=True, dtype=torch.float16, device_type=device.type):
        pred_top = sliding_window_inference(
            top_half, roi_size=roi_size, sw_batch_size=sw_batch_size,
            predictor=model, sw_device=device, device='cpu', **kwargs
        )
        del top_half
        gc.collect()
        torch.cuda.empty_cache()

        pred_bottom = sliding_window_inference(
            bottom_half, roi_size=roi_size, sw_batch_size=sw_batch_size,
            predictor=model, sw_device=device, device='cpu', **kwargs
        )
        
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
            if overlap > 0:
                # Average overlap region
                out[..., mid - overlap:mid] = (out[..., mid - overlap:mid] + t_bottom[..., :overlap]) / 2.0
                out[..., mid:] = t_bottom[..., overlap:]
            else:
                out[..., mid:] = t_bottom

            del t_bottom
            gc.collect()
            torch.cuda.empty_cache()
            merged.append(out)

    return tuple(merged)

def inference_step(args, model, data_sample, transform, device):
    with torch.no_grad(), torch.amp.autocast(enabled=True, dtype=torch.float16, device_type=device.type):
        
        output = memory_efficient_inference_with_overlap(args, model, data_sample['image'], args.patch_size, device=device, overlap=0.5, 
                                                        blend_mode="gaussian", sigma_scale=0.125, cast_dtype=torch.float16)
        
        #unpack output
        (multiclass_segmentation, dist_pred, pulp_segmentation) = output # all on GPU
        binary_mask = torch.where(multiclass_segmentation >= 1, 1, 0).to(torch.uint8)
        markers = torch.where(dist_pred > 0.5, 1, 0).to(torch.uint8)
        #CPU
        dist_pred_cpu = dist_pred.squeeze().cpu()
        multiclass_segmentation_cpu = multiclass_segmentation.squeeze().cpu()
        binary_mask_cpu =  binary_mask.squeeze().cpu()
        markers_cpu = markers.squeeze().cpu()
        #post_process
        pred_multiclass = deep_watershed_with_voting(dist_pred_cpu.numpy(), 
                                                    multiclass_segmentation_cpu.numpy(), 
                                                    binary_mask_cpu.numpy(), 
                                                    markers_cpu.numpy())
        # pred_multiclass = deep_watershed_with_voting_optimized(to_numpy(dist_pred_cpu.numpy()), 
        #                                                        to_numpy(multiclass_segmentation_cpu.numpy()), 
        #                                                        to_numpy(binary_mask_cpu.numpy()), 
        #                                                        to_numpy(markers_cpu.numpy()))
        pred_multiclass_gpu = torch.from_numpy(pred_multiclass).to(device).long() # H,W,D
        # ---- Merge pulp and remap labels ----``
        pred_with_pulp = merge_pulp_into_teeth_torch(pred_multiclass_gpu.squeeze(), pulp_segmentation.squeeze(), pulp_class=50).to(torch.int32)  # H,W,D
        remapped = remap_labels_torch(pred_with_pulp, pred_to_challange_map)
        
        # save results to disk
        data_sample['pulp'] = MetaTensor(pulp_segmentation)
        data_sample['dist'] = MetaTensor(dist_pred)
        data_sample['mlt'] = MetaTensor(remapped.unsqueeze(0).unsqueeze(0)) # HERE IS CHANGED INPUT
        inverted_prediction_mlt = [transform.post_inference_transform_no_dir(i) for i in decollate_batch(data_sample)] 
        # inverted_prediction = transform.post_inference_transform(data_sample) 
        transform.save_inference_output(inverted_prediction_mlt[0])
        
        return remapped

def main():

    is_ram_efficient_inference = True
    is_new_inference = True

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
    input_image = [
        # {"image": "data/imagesTr/ToothFairy3F_001_0000.nii.gz"},
        # {"image": "data/imagesTr/ToothFairy3F_005_0000.nii.gz"},
        {"image": "data/imagesTr/ToothFairy3F_009_0000.nii.gz"}, #VAL
        {"image": "data/imagesTr/ToothFairy3F_013_0000.nii.gz"}, #VAL
        # {"image": "data/imagesTr/ToothFairy3F_010_0000.nii.gz"},
        {"image": "data/imagesTr/ToothFairy3P_059_0000.nii.gz"}, #VAL
        {"image": "data/imagesTr/ToothFairy3P_095_0000.nii.gz"}, #VAL
        # {"image": "data/imagesTr/ToothFairy3P_381_0000.nii.gz"},
        # {"image": "data/imagesTr/ToothFairy3P_386_0000.nii.gz"},
        # {"image": "data/imagesTr/ToothFairy3P_391_0000.nii.gz"},
        {"image": "data/imagesTr/ToothFairy3S_0001_0000.nii.gz"}, #VAL
        {"image": "data/imagesTr/ToothFairy3S_0014_0000.nii.gz"}, #VAL
        # {"image": "data/imagesTr/ToothFairy3S_0005_0000.nii.gz"},
        # {"image": "data/imagesTr/ToothFairy3S_0010_0000.nii.gz"},
        # {"image": "data/imagesTr/ToothFairy3S_0015_0000.nii.gz"},
    ]
    
    dataset = Dataset(data=input_image, transform=transform.inference_preprocessing)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
    # data_sample = next(iter(dataloader))
    
    for data_sample in dataloader:
        print(f"Processing image: {data_sample['image'].meta['filename_or_obj'][0]}, shape: {data_sample['image'].shape}")
        # max_voxels = 10*1024**3 // (48 * 2)
        # z_slab_max = max_voxels // (data_sample["image"].shape[-3] * data_sample["image"].shape[-2])
        # pz = args.patch_size[2]
        # if z_slab_max < pz:
        #     print("pixdim should be lower, cannot fit in RAM single patch_size z layer")
    
        # data_sample = transform.inference_preprocessing(input_image)
        
        with torch.amp.autocast(enabled=True, dtype=torch.float16, device_type=device.type):
            with torch.no_grad():
                #model
                model = DWNet(spatial_dims=3, in_channels=1, out_channels=args.out_channels, act=args.activation, norm=args.norm,
                              bias=False, backbone_name=args.backbone_name, configuration='DIST_PULP')
                model.load_state_dict(torch.load('checkpoints/checkpoints/silent_pie_5061/model_epoch_300.pth',
                                                map_location=device, weights_only=True)['model_state_dict'], strict=False)
                model = model.to(device)
                model.eval()
                
                print("Start inference...")
                start_time = time.time()
                if is_ram_efficient_inference:
                    if is_new_inference:
                        multiclass_probs, dist_probs, pulp_probs, weights_accumulator = \
                            memory_efficient_inference_final(args, model, data_sample["image"], args.patch_size, device=device, overlap=0.25, blend_mode="gaussian",
                                                        sigma_scale=0.125, cast_dtype=torch.float16, dist_map_cast_dtype=torch.float16)
                        model = free(model)
                        #multiclass_probs = multiclass_probs.to(device)
                        # multiclass_pred = (multiclass_probs / weights_accumulator.cpu()).argmax(dim=1).squeeze().to(device=device, dtype=torch.uint8) 
                        #multiclass_pred = (multiclass_probs / weights_accumulator).argmax(dim=1).squeeze().to(device='cpu', dtype=torch.uint8) #min weights_acc is 1e-3 - min of gaussian map
                        multiclass_pred = multiclass_probs.squeeze() # TODO fix it 
                        # print((multiclass_probs / weights_accumulator).float().sum())
                        # multiclass_probs = free(multiclass_probs)
                        
                        binary_mask = torch.where(multiclass_pred >= 1, 1, 0).to(torch.int8).squeeze().cpu()
                        dist_pred =  nn.Threshold(1e-3, 0)(dist_probs / weights_accumulator).squeeze().cpu()
                        dist_probs = free(dist_probs)
                        
                        markers = torch.where(dist_pred > 0.5, 1, 0).to(torch.int8).squeeze().cpu()
                        pulp_segmentation = (pulp_probs / weights_accumulator > 0.5).to(torch.uint8).squeeze()
                        pulp_probs = free(pulp_probs)
                        #time
                        network_pass_time = time.time() - start_time
                        print(f"Network pass time: {network_pass_time:.2f} seconds")
                    else:
                        multiclass_segmentation, dist_pred, pulp_segmentation = \
                            memory_efficient_inference_with_overlap(args, model, data_sample["image"], args.patch_size, device=device, overlap=0.25,
                                                                    blend_mode="gaussian", sigma_scale=0.125, cast_dtype=torch.float16)                        
                        markers = torch.where(dist_pred > 0.5, 1, 0).to(torch.int8).squeeze().cpu()
                        dist_pred = dist_pred.squeeze().cpu()
                        pulp_segmentation = pulp_segmentation.squeeze()
                        binary_mask = torch.where(multiclass_segmentation >= 1, 1, 0).to(torch.int8).squeeze().cpu()
                        multiclass_pred = multiclass_segmentation.squeeze().cpu()
                        #time
                        network_pass_time = time.time() - start_time
                        print(f"Network pass time: {network_pass_time:.2f} seconds")
                else:
                    (multiclass_segmentation, dist_pred, pulp_segmentation) =  \
                        sliding_window_inference(data_sample["image"], roi_size=args.patch_size, sw_batch_size=1, predictor=model,
                                                 overlap=0.25, sw_device=device, device=device, mode='gaussian', sigma_scale=0.125, padding_mode='constant', cval=0, progress=False)
                    multiclass_pred = multiclass_segmentation.argmax(dim=1, keepdim=True).squeeze().cpu()
                    dist_pred = nn.Threshold(1e-3, 0)(torch.sigmoid(dist_pred)).to(torch.float16).squeeze().cpu()
                    pulp_segmentation = (torch.sigmoid(pulp_segmentation) > 0.5).to(torch.int8).squeeze()
                    binary_mask = torch.where(multiclass_pred >= 1, 1, 0).to(torch.int8).squeeze().cpu()
                    markers = torch.where(dist_pred > 0.5, 1, 0).to(torch.int8).squeeze().cpu()
                    #time
                    network_pass_time = time.time() - start_time
                    print(f"Network pass time: {network_pass_time:.2f} seconds")
        
        post_proc_time = time.time()
        pred_multiclass = deep_watershed_with_voting_optimized(dist_pred.numpy(), 
                                                               multiclass_pred.numpy(), 
                                                               binary_mask.numpy(),
                                                               markers.numpy())
        post_proc_time = time.time() - post_proc_time
        print(f"Postproc pass time: {post_proc_time:.2f} seconds")
        
        print(np.unique(pred_multiclass))
        
        pred_multiclass_gpu = torch.from_numpy(pred_multiclass).to(dtype=torch.long, device=device)
        pred_with_pulp = merge_pulp_into_teeth_torch(pred_multiclass_gpu, pulp_segmentation, pulp_class=50).to(torch.int32) 
        remapped = remap_labels_torch(pred_with_pulp, pred_to_challange_map)
            ### invert transforms, eg. padding
            # prediction = transform.post_inference_transform({"pred": MetaTensor(remapped.unsqueeze(0)), "image": data_sample["image"][0]})["pred"] # B,H,W,D

        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time:.2f} seconds")
        
        #SAVE RESULTS TO DISK TO PREVIEW
        data_sample['mlt'] = MetaTensor(pred_multiclass_gpu.unsqueeze(0).unsqueeze(0))
        data_sample['pulp'] = MetaTensor(pulp_segmentation.unsqueeze(0).unsqueeze(0))
        data_sample['dist'] = MetaTensor(dist_pred.unsqueeze(0).unsqueeze(0))
        data_sample['final'] = MetaTensor(remapped.unsqueeze(0).unsqueeze(0))
        
        inverted_prediction_mlt = [transform.post_inference_transform_no_dir(i) for i in decollate_batch(data_sample)] 
        save_transform = SaveMultipleKeysD(
            keys=['mlt', 'pulp', 'dist', 'final'],
            output_dir='output/inference_results',
            output_postfixes=['mlt', 'pulp', 'dist', 'final'],
            separate_folder=False,
            output_dtype=[np.uint8, np.uint8, np.float32, np.uint8]
        )
        save_transform(inverted_prediction_mlt[0])
        
        #manual save - direct, no inversion, debug inversion
        # save_nifti(pred_multiclass_gpu_clone, path='output', filename="manual_mlt.nii.gz", pixdim=0.2, dtype=np.uint8)
        # save_nifti(remapped, path='output', filename="manual_final.nii.gz", pixdim=0.2, dtype=np.uint8)
        # save_nifti(dist_pred, path='output', filename="manual_dist.nii.gz", pixdim=0.2, dtype=np.float32)
        
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
    start_time_epoch = time.time()
    main()
    inference_time=time.time() - start_time_epoch
    print(f"Inference took: {inference_time:.2f}s.")