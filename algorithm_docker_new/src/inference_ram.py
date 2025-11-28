import torch
import gc
import os
import time
import numpy as np
from typing import Tuple, Optional, Sequence
from monai.data.utils import compute_importance_map, dense_patch_slices
from monai.data.meta_tensor import MetaTensor

from .model import DWNet
from .inference_utils import merge_pulp_into_teeth_torch, remap_labels_torch, pred_to_challange_map

def free_tensor(t):
    del t
    gc.collect()
    torch.cuda.empty_cache()

def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: Sequence[float]
) -> tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError(f"len(image_size) {len(image_size)} different from spatial dims {num_spatial_dims}.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError(f"len(roi_size) {len(roi_size)} different from spatial dims {num_spatial_dims}.")

    scan_interval = []
    for i, o in zip(range(num_spatial_dims), overlap):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - o))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)
    
def memory_efficient_inference_final(
    args,
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    patch_size: Tuple[int, int, int],
    device: Optional[torch.device] = None,
    device_aux: str = "gpu",
    device_accum: str = "cpu",
    overlap: float = 0.25,
    blend_mode: str = "gaussian",
    sigma_scale: float = 0.125,
    cast_dtype: torch.dtype = torch.float16,
    dist_map_cast_dtype=torch.float32,
    normalize_to_fp32: bool = False,
    is_memfile : bool = True,
    is_memfile_torch : bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Memory-efficient sliding-window inference.

    """

    device = device or input_tensor.device
    if device_aux == "gpu":
        device_aux = device 
    else:
        device_aux = torch.device("cpu")
    
    if device_accum == "gpu":
        device_accum = device
    else:
        device_accum = torch.device("cpu")

    _, _, H, W, D = input_tensor.shape  # input layout: (B=1, C_in, H, W, D)
    
    #Setup sliding window patches
    overlap = overlap if isinstance(overlap, (tuple, list)) else (overlap,) * len(patch_size)
    scan_interval = _get_scan_interval((H, W, D), patch_size, len(patch_size), overlap)
    slices = dense_patch_slices((H, W, D), patch_size, scan_interval)

    #Prepare z starts for slab flush 
    def _starts(length, patch, stride):
        if length <= patch:
            return [0]
        starts = list(range(0, length - patch + 1, stride))
        if starts[-1] + patch < length:
            starts.append(length - patch)
        return starts

    z_starts = _starts(D, patch_size[2], scan_interval[2])

    slices_by_z = {}
    for s in slices:
        z0 = s[2].start
        slices_by_z.setdefault(z0, []).append(s)

    pz = patch_size[2]

    # (1,1,ph,pw,pz)
    importance_map = compute_importance_map(
        patch_size, mode=blend_mode, sigma_scale=sigma_scale, device=device, dtype=cast_dtype
    )[None, None]  

    #ACCUMULATORS
    #cpu
    # pred_seg = torch.zeros((1, H, W, D), dtype=torch.uint8, device="cpu")
    pred_seg = torch.empty((1, H, W, D), dtype=torch.uint8, device=device_accum)
    pred_seg.zero_()

    if is_memfile:
        filename = "/tmp/slab_probs.dat"
        shape = (1, args.out_channels, H, W, pz)
        numel = args.out_channels * H * W * pz
        if not os.path.exists(filename):
            f = open(filename, "wb")
            f.seek(numel * 2 - 1)  # float16 = 2 bytes per element
            f.write(b"\0")
            f.close()
        if is_memfile_torch:
            slab_probs = torch.from_file(filename, dtype=torch.float16, size=numel, device=device_accum)
            slab_probs = slab_probs.view(shape)  
        else:
            #file numpy memmap
            shape = (1, args.out_channels, H, W, pz)
            slab_probs_fnpm = np.memmap(filename, dtype=np.float16, mode='w+', shape=shape)
    else:      
        slab_probs = torch.empty((1, args.out_channels, H, W, pz), dtype=cast_dtype, device=device_accum)
        slab_probs.zero_()
    #gpu or cpu
    weights_accumulator = torch.zeros((1, 1, H, W, D), dtype=cast_dtype, device=device_aux)
    # dist_probs = torch.zeros((1, 1, H, W, D), dtype=dist_map_cast_dtype, device=device_aux)
    pulp_probs = torch.zeros((1, 1, H, W, D), dtype=cast_dtype, device=device_aux)

    # Initialize slab_start at the first z_start
    slab_start = z_starts[0]
    n_z = len(z_starts)
    
    print("start inference over slabs")

    for z_idx, z0 in enumerate(z_starts):
        # Process all XY patches that start at z = z0
        for s in slices_by_z[z0]:
            # logits, dist_patch, pulp_patch = model(input_tensor[:, :, s[0], s[1], s[2]])
            logits, pulp_patch = model(input_tensor[:, :, s[0], s[1], s[2]])
            
            weighted_probs = torch.softmax(logits, dim=1).to(cast_dtype) * importance_map
            free_tensor(logits)
            # weighted_dist = torch.sigmoid(dist_patch).to(dist_map_cast_dtype) * importance_map
            weighted_pulp = torch.sigmoid(pulp_patch).to(cast_dtype) * importance_map

            weighted_probs = weighted_probs.to(device_accum)          
            if device_aux != device:
                # weighted_dist = weighted_dist.to(device_aux)
                weighted_pulp = weighted_pulp.to(device_aux)
                importance_map = importance_map.to(device_aux)

            # Accumulate into slab buffer and full-volume 1-channel maps
            if is_memfile:
                slab_slice = torch.from_numpy(slab_probs_fnpm[:, :, s[0], s[1], :])
                slab_slice += weighted_probs #CPU
                # slab_probs_fnpm.flush()
            else:
                slab_probs[:, :, s[0], s[1], :] += weighted_probs #CPU
            weights_accumulator[:, :, s[0], s[1], s[2]] += importance_map
            # dist_probs[:, :, s[0], s[1], s[2]] += weighted_dist
            pulp_probs[:, :, s[0], s[1], s[2]] += weighted_pulp
            
            free_tensor(weighted_probs)

        # Compute how many leading slices are now safe to flush.
        if z_idx < (n_z - 1):
            next_z0 = z_starts[z_idx + 1]
            flush_end = z0 + (next_z0 - z0)
        else:
            # Last z block: flush everything remaining
            flush_end = D

        flush_end_clamped = min(flush_end, slab_start + pz, D)
        flush_len = flush_end_clamped - slab_start
        if flush_len > 0:
            weights_slice = weights_accumulator[:, :, :, :, slab_start: slab_start + flush_len].to(device_accum)
            # Compute how many slices from the tail of the patch we must carry forward
            carry_len = max(0, pz - flush_len)
            
            # Clone only the part that must be preserved for the next slab (the overlap tail)
            overlap_backup = None
            if carry_len > 0 and flush_end_clamped != D:
                if is_memfile:
                    overlap_backup = torch.from_numpy(slab_probs_fnpm[:, :, :, :, flush_len:flush_len+carry_len]).clone()
                else:
                    overlap_backup = slab_probs[:, :, :, :, flush_len : flush_len + carry_len].clone()

            # Compute probabilities for flush
            probs_slice = torch.from_numpy(slab_probs_fnpm[:, :, :, :, :flush_len]) if is_memfile else slab_probs[:, :, :, :, :flush_len]
            if normalize_to_fp32:
                probs_slice = (probs_slice.float() / weights_slice.float()).to(cast_dtype)
            else:
                probs_slice = (probs_slice / weights_slice).to(cast_dtype)

            #assign prediction to final seg map
            pred_seg[:, :, :, slab_start: slab_start + flush_len] = torch.argmax(probs_slice, dim=1).to(torch.uint8)

            # If this is the final flush break
            if flush_end_clamped == D:
                if is_memfile:
                    del slab_probs_fnpm
                    os.remove(filename)
                else:
                    slab_probs.zero_() # TODO is it necessary?
                slab_start += flush_len
                break

            # Restore the overlap from backup at the beginning of slab buffer
            if is_memfile:
                if carry_len > 0:
                    slab_probs_fnpm[:, :, :, :, :carry_len] = overlap_backup
                    slab_probs_fnpm[:, :, :, :, carry_len:] = 0
                    del overlap_backup
                else:
                    slab_probs_fnpm[:, :, :, :, :flush_len] = 0
            else:
                if carry_len > 0:
                    slab_probs[:, :, :, :, :carry_len] = overlap_backup
                    slab_probs[:, :, :, :, carry_len:] = 0
                    del overlap_backup
                else:
                    slab_probs.zero_()
       
            slab_start += flush_len
            
    print("Inference over slabs completed.")
    # This should never trigger; retained as a safeguard for misaligned patch sets
    if slab_start < D:
        print("WARNING: Final slab flush triggered. This indicates logic bug in slab tracking.")
        # Remaining length
        remaining_len = min(pz, D - slab_start)
        if remaining_len > 0:
            probs_slice = slab_probs[:, :, :, :, :remaining_len]
            weights_slice = weights_accumulator[:, :, :, :, slab_start: slab_start + remaining_len].to(device_accum)
            if normalize_to_fp32:
                probs_slice = (probs_slice.float() / weights_slice.float()).to(cast_dtype)
            else:
                probs_slice /= weights_slice
            pred_slice = torch.argmax(probs_slice, dim=1).to(torch.uint8)
            pred_seg[:, :, :, slab_start: slab_start + remaining_len] = pred_slice

    # return pred_seg, dist_probs, pulp_probs, weights_accumulator
    return pred_seg.squeeze(), pulp_probs, weights_accumulator


def run_inference(input_tensor, args, device, transform) -> np.ndarray:
    model = DWNet(spatial_dims=3, in_channels=1, out_channels=args.out_channels, act=args.activation, norm=args.norm,
                    bias=False, backbone_name=args.backbone_name, configuration=args.configuration)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device, weights_only=True)['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    start_time = time.time()
    with torch.inference_mode(), torch.amp.autocast(enabled=True, dtype=torch.float16, device_type=device.type):
        multiclass_pred, pulp_segmentation, weights_accumulator = \
                        memory_efficient_inference_final(args, model, input_tensor["image"], args.patch_size, device=device, device_accum='gpu', overlap=0.1, blend_mode="gaussian",
                                                         sigma_scale=0.125, cast_dtype=torch.float16, dist_map_cast_dtype=torch.float16, is_memfile=False)
        model = free_tensor(model)
        pulp_segmentation.div_(weights_accumulator) 
        pulp_segmentation.gt_(0.5) 
        pulp_segmentation = pulp_segmentation.to(torch.int8).squeeze()
    
    print("inference time: %.2f seconds" % (time.time() - start_time))        
    print("merging pulp segmentation into teeth labels and remapping to challenge labels")    

    multiclass_pred = multiclass_pred.to(dtype=torch.int8, device=device)
    pred_with_pulp = merge_pulp_into_teeth_torch(multiclass_pred, pulp_segmentation.to(device), pulp_class=50)
    remapped = remap_labels_torch(pred_with_pulp.to(torch.int32), pred_to_challange_map)
    
    prediction = transform.post_inference_transform({"pred": MetaTensor(remapped.unsqueeze(0)), "image": input_tensor["image"][0]})["pred"] # B,H,W,D
    return prediction.squeeze().cpu().numpy().astype(np.int8) # H,W,D