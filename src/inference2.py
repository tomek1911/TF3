import torch
import time
import argparse
import yaml
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np
import gc
import SimpleITK as sitk
import nibabel as nib
from nibabel.orientations import aff2axcodes

# from monai.inferers import sliding_window_inference
from copy import copy
from monai.data import DataLoader, decollate_batch
from monai.data.dataset import Dataset
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import compute_importance_map, BlendMode
from monai.inferers.utils import compute_importance_map

from deep_watershed import deep_watershed_with_voting, deep_watershed_with_voting_optimized
from inference_utils import merge_pulp_into_teeth_torch, remap_labels_torch, pred_to_challange_map, save_nifti
from transforms import Transforms
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

        # max_voxels = 10*1024**3 // (48 * 2)
        # z_slab_max = max_voxels // (data_sample["image"].shape[-3] * data_sample["image"].shape[-2])
        # pz = args.patch_size[2]
        # if z_slab_max < pz:
        #     print("pixdim should be lower, cannot fit in RAM single patch_size z layer")
        
        # data_sample = transform.inference_preprocessing(input_image)
        model = DWNet(spatial_dims=3, in_channels=1, out_channels=args.out_channels, act=args.activation, norm=args.norm,
                    bias=False, backbone_name=args.backbone_name, configuration='DIST_PULP')
        model.load_state_dict(torch.load('checkpoints/checkpoints/silent_pie_5061/model_epoch_380.pth',
                                        map_location=device, weights_only=True)['model_state_dict'], strict=False)
        model = model.to(device)
        model.eval()
        
        with torch.no_grad(), torch.amp.autocast(enabled=True, dtype=torch.float16, device_type=device.type):
            ## SLOW - MEMORY EFFICIENT
            # output = memory_efficient_inference_with_overlap(args, model, data_sample["image"], args.patch_size, device=device, overlap=0.25, 
            #                                                  blend_mode="gaussian", sigma_scale=0.125, cast_dtype=torch.float16)
            #(multiclass_segmentation, dist_pred, pulp_segmentation) = output
            
            ## FAST - MEMORY HUNGRY
            output = sliding_window_inference(data_sample["image"], roi_size=args.patch_size, sw_batch_size=1, predictor=model, 
                                            overlap=0.25, sw_device=device, device=device, mode='gaussian', sigma_scale=0.125,
                                            padding_mode='constant', cval=0, progress=False)
            (multiclass_segmentation, dist_pred, pulp_segmentation) = output
            #preds
            multiclass_segmentation = multiclass_segmentation.argmax(dim=1, keepdim=True).to(dtype=torch.uint8, device=device)
            dist_pred = nn.Threshold(1e-3, 0)(torch.sigmoid(dist_pred)).to(torch.float16)
            pulp_segmentation = (torch.sigmoid(pulp_segmentation) > 0.5).to(torch.uint8)
            
            binary_mask = torch.where(multiclass_segmentation >= 1, 1, 0).to(torch.uint8) # values: 0-1
            markers = torch.where(dist_pred > 0.5, 1, 0).to(torch.uint8) # values: 0-1
        
        dist_pred_cpu = dist_pred.squeeze().cpu()
        multiclass_segmentation_cpu = multiclass_segmentation.squeeze().cpu()
        binary_mask_cpu =  binary_mask.squeeze().cpu()
        markers_cpu = markers.squeeze().cpu()
        
        pred_multiclass = deep_watershed_with_voting_optimized(dist_pred_cpu.numpy(), 
                                                            multiclass_segmentation_cpu.numpy(), 
                                                            binary_mask_cpu.numpy(),
                                                            markers_cpu.numpy())
        with torch.no_grad():
            pred_multiclass_gpu = torch.from_numpy(pred_multiclass).long().to(device)
            pred_multiclass_gpu_clone = pred_multiclass_gpu.clone()
            pred_with_pulp = merge_pulp_into_teeth_torch(pred_multiclass_gpu.squeeze(), pulp_segmentation.squeeze(), pulp_class=50).to(torch.int32) 
            remapped = remap_labels_torch(pred_with_pulp, pred_to_challange_map)
            ### invert transforms, eg. padding
            # prediction = transform.post_inference_transform({"pred": MetaTensor(remapped.unsqueeze(0)), "image": data_sample["image"][0]})["pred"] # B,H,W,D

        #SAVE RESULTS TO DISK TO PREVIEW
        data_sample['pulp'] = MetaTensor(pulp_segmentation)
        data_sample['dist'] = MetaTensor(dist_pred)
        data_sample['mlt'] = MetaTensor(pred_multiclass_gpu_clone.unsqueeze(0).unsqueeze(0))
        data_sample['final'] = MetaTensor(remapped.unsqueeze(0).unsqueeze(0))
        
        inverted_prediction_mlt = [transform.post_inference_transform_no_dir(i) for i in decollate_batch(data_sample)] 
        transform.save_inference_output(inverted_prediction_mlt[0])
        
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