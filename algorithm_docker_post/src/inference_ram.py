import dataclasses
import torch
import gc
import os
import time
import numpy as np
from typing import Tuple, Optional, Sequence
from monai.data.utils import compute_importance_map, dense_patch_slices
from monai.data.meta_tensor import MetaTensor

from .model import DWNet
from .inference_utils import remap_labels_torch, pred_to_challange_map

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

    # Zero-pad Z to at least pz if the volume is shallower than one patch.
    # This is a self-contained "transform + inverse": we record d_orig and crop
    # pred_seg back to it before returning so callers see the original Z extent.
    pz_target = patch_size[2]
    d_orig = D
    if D < pz_target:
        pad_d = pz_target - D
        input_tensor = torch.nn.functional.pad(input_tensor, (0, pad_d))  # pad last dim only
        D = pz_target
        print(f"[inference] Z={d_orig} < patch_z={pz_target}: zero-padded to {D}, will crop back on return.")

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
    # pred_seg is uint8 with no arithmetic — lives on CPU (0.25 GB, no fp16 concern)
    pred_seg = torch.zeros((1, H, W, D), dtype=torch.uint8, device='cpu')
    pred_seg.zero_()

    # --- Auto-select number of H-stripes based on VRAM budget ---
    # Slab bytes = (C + 1) channels (probs + weights) * H * W * pz * bytes_per_element
    # We keep H un-split for num_stripes=1, else split into 2 or 4 equal stripes.
    slab_vram_budget_gb = getattr(args, 'slab_vram_budget_gb', 6.0)
    dtype_bytes = torch.empty([], dtype=cast_dtype).element_size()  # 2 for fp16, 4 for fp32
    slab_bytes_full = (args.out_channels + 1) * H * W * pz * dtype_bytes
    budget_bytes = slab_vram_budget_gb * 1024**3
    if slab_bytes_full <= budget_bytes:
        num_stripes = 1
    elif slab_bytes_full / 2 <= budget_bytes:
        num_stripes = 2
    else:
        num_stripes = 4
    print(f"Slab size: {slab_bytes_full/1024**3:.2f} GB, budget: {slab_vram_budget_gb:.1f} GB → using {num_stripes} stripe(s)")

    # Compute stripe H boundaries
    stripe_boundaries = []
    base = H // num_stripes
    rem  = H % num_stripes
    h_cursor = 0
    for i in range(num_stripes):
        h_end = h_cursor + base + (1 if i < rem else 0)
        stripe_boundaries.append((h_cursor, h_end))
        h_cursor = h_end

    use_stripes = num_stripes > 1

    if not use_stripes:
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
            slab_probs = torch.zeros((1, args.out_channels, H, W, pz), dtype=cast_dtype, device=device_accum)
            slab_probs.zero_()
        # slab-sized weight buffer — same z-window as slab_probs, avoids full-volume (H,W,D) allocation (~0.5 GB)
        weights_slab = torch.zeros((1, 1, H, W, pz), dtype=cast_dtype, device=device)
    else:
        # In stripe mode, carry_len is the fixed number of z-slices that overlap between
        # consecutive z-groups (non-last groups only). carry_len = pz - scan_interval_z.
        # This is constant for all non-last z-groups.
        carry_len_fixed = pz - scan_interval[2]  # e.g. 128 - 115 = 13
        # Persistent per-stripe carry buffers (probs + weights), initialised to zero.
        # Size: (1, C, stripe_h, W, carry_len) — ~24 MB total for carry_len=13, negligible.
        stripe_carry_probs   = [
            torch.zeros((1, args.out_channels, sh_end - sh_start, W, carry_len_fixed), dtype=cast_dtype, device=device)
            for (sh_start, sh_end) in stripe_boundaries
        ]
        stripe_carry_weights = [
            torch.zeros((1, 1, sh_end - sh_start, W, carry_len_fixed), dtype=cast_dtype, device=device)
            for (sh_start, sh_end) in stripe_boundaries
        ]

    # Initialize slab_start at the first z_start
    slab_start = z_starts[0]
    n_z = len(z_starts)
    
    print("start inference over slabs")

    for z_idx, z0 in enumerate(z_starts):

        is_last_z = (z_idx == n_z - 1)

        # Compute flush window for this z-group (same formula for both modes)
        if not is_last_z:
            next_z0 = z_starts[z_idx + 1]
            flush_end = z0 + (next_z0 - z0)   # == next_z0
        else:
            flush_end = D
        flush_end_clamped = min(flush_end, slab_start + pz, D)
        flush_len = flush_end_clamped - slab_start
        # carry_len: how many z-slices from the tail of the current slab window must be
        # carried into the next z-group. Zero for the last z-group.
        carry_len = max(0, pz - flush_len) if not is_last_z else 0

        if use_stripes:
            # --- STRIPE MODE ---
            # Process each H-stripe independently to cap peak VRAM.
            # Per-stripe carry buffers (stripe_carry_probs/weights) are persistent across
            # z-groups so that gaussian-weighted z-overlap is correctly handled.
            for stripe_idx, (sh_start, sh_end) in enumerate(stripe_boundaries):
                stripe_h = sh_end - sh_start

                # Allocate full-size accumulator for this stripe (freed at end of stripe loop)
                slab_stripe    = torch.zeros((1, args.out_channels, stripe_h, W, pz), dtype=cast_dtype, device=device)
                weights_stripe = torch.zeros((1, 1,                 stripe_h, W, pz), dtype=cast_dtype, device=device)

                # Warm-start: copy carry from previous z-group into the front of the slab.
                # carry_len_fixed slices at positions [0:carry_len_fixed] already contain
                # the overlap contribution from the previous z-group's patches.
                if carry_len_fixed > 0:
                    slab_stripe[:, :, :, :, :carry_len_fixed]    = stripe_carry_probs[stripe_idx]
                    weights_stripe[:, :, :, :, :carry_len_fixed] = stripe_carry_weights[stripe_idx]

                # Accumulate all patches that touch this stripe
                for s in slices_by_z[z0]:
                    ph_start = s[0].start
                    ph_end   = s[0].stop
                    inter_start = max(ph_start, sh_start)
                    inter_end   = min(ph_end,   sh_end)
                    if inter_start >= inter_end:
                        continue  # patch doesn't touch this stripe

                    logits, _ = model(input_tensor[:, :, s[0], s[1], s[2]])
                    weighted_probs = torch.softmax(logits, dim=1).to(cast_dtype) * importance_map
                    del logits

                    # Row offsets: within the patch, and within the stripe buffer
                    loc_h_s = inter_start - ph_start
                    loc_h_e = inter_end   - ph_start
                    str_h_s = inter_start - sh_start
                    str_h_e = inter_end   - sh_start

                    slab_stripe[:, :, str_h_s:str_h_e, s[1], :]    += weighted_probs[:, :, loc_h_s:loc_h_e, :, :]
                    weights_stripe[:, :, str_h_s:str_h_e, s[1], :] += importance_map[:, :, loc_h_s:loc_h_e, :, :]

                    del weighted_probs
                    torch.cuda.empty_cache()

                # Save carry for next z-group (tail carry_len_fixed slices of the slab).
                # Done before flush so we don't need the full buffer afterwards.
                if not is_last_z and carry_len_fixed > 0:
                    stripe_carry_probs[stripe_idx].copy_(
                        slab_stripe[:, :, :, :, flush_len:flush_len + carry_len_fixed]
                    )
                    stripe_carry_weights[stripe_idx].copy_(
                        weights_stripe[:, :, :, :, flush_len:flush_len + carry_len_fixed]
                    )

                # Flush: normalise the flush_len slices and write argmax to pred_seg
                if flush_len > 0:
                    p_sl = slab_stripe[:, :, :, :, :flush_len]
                    w_sl = weights_stripe[:, :, :, :, :flush_len]
                    p_sl = p_sl.clone()   # clone so div_ doesn't alias the slab buffer
                    p_sl.div_(w_sl)
                    pred_seg[:, sh_start:sh_end, :, slab_start:slab_start + flush_len] = (
                        torch.argmax(p_sl, dim=1).to(torch.uint8).cpu()
                    )
                    del p_sl

                del slab_stripe, weights_stripe
                torch.cuda.empty_cache()

            # Advance slab_start after all stripes are done
            if flush_len > 0:
                slab_start += flush_len
            if flush_end_clamped == D:
                break

        else:
            # --- STANDARD SLAB MODE (num_stripes == 1) ---
            for s in slices_by_z[z0]:
                logits, _ = model(input_tensor[:, :, s[0], s[1], s[2]])

                weighted_probs = torch.softmax(logits, dim=1).to(cast_dtype) * importance_map
                # del immediately — free_tensor(x) only deletes its local param, not the caller's reference
                del logits

                # Accumulate into slab probability buffer and slab weight buffer
                if is_memfile:
                    slab_slice = torch.from_numpy(slab_probs_fnpm[:, :, s[0], s[1], :])
                    slab_slice += weighted_probs.cpu()
                else:
                    slab_probs[:, :, s[0], s[1], :] += weighted_probs
                weights_slab[:, :, s[0], s[1], :] += importance_map

                del weighted_probs
                torch.cuda.empty_cache()

            if flush_len > 0:
                weights_slice = weights_slab[:, :, :, :, :flush_len]

                # Compute probabilities for flush.
                # div_ in-place avoids allocating a temporary fp32 tensor 
                probs_slice = torch.from_numpy(slab_probs_fnpm[:, :, :, :, :flush_len]) if is_memfile else slab_probs[:, :, :, :, :flush_len]
                probs_slice.div_(weights_slice)

                pred_seg[:, :, :, slab_start: slab_start + flush_len] = torch.argmax(probs_slice, dim=1).to(torch.uint8).cpu()

                # If this is the final flush, clean up and break
                if flush_end_clamped == D:
                    if is_memfile:
                        del slab_probs_fnpm
                        os.remove(filename)
                    else:
                        slab_probs.zero_()
                        weights_slab.zero_()
                    slab_start += flush_len
                    break

                # Shift carry portion to front of slab buffers in-place.
                # Safe: with overlap=0.1, pz=128: flush_len=115, carry_len=13
                # → src [115:128] and dest [0:13] don't overlap.
                if is_memfile:
                    if carry_len > 0:
                        slab_probs_fnpm[:, :, :, :, :carry_len] = slab_probs_fnpm[:, :, :, :, flush_len:flush_len+carry_len]
                        slab_probs_fnpm[:, :, :, :, carry_len:] = 0
                    else:
                        slab_probs_fnpm[:, :, :, :, :] = 0
                else:
                    if carry_len > 0:
                        slab_probs[:, :, :, :, :carry_len].copy_(slab_probs[:, :, :, :, flush_len:flush_len+carry_len])
                        slab_probs[:, :, :, :, carry_len:].zero_()
                        weights_slab[:, :, :, :, :carry_len].copy_(weights_slab[:, :, :, :, flush_len:flush_len+carry_len])
                        weights_slab[:, :, :, :, carry_len:].zero_()
                    else:
                        slab_probs.zero_()
                        weights_slab.zero_()

                slab_start += flush_len

    print("Inference over slabs completed.")
    if not use_stripes and slab_start < D:
        # This should never trigger; retained as a safeguard for misaligned patch sets
        print("WARNING: Final slab flush triggered. This indicates logic bug in slab tracking.")
        remaining_len = min(pz, D - slab_start)
        if remaining_len > 0:
            probs_slice = slab_probs[:, :, :, :, :remaining_len]
            weights_slice = weights_slab[:, :, :, :, :remaining_len]
            probs_slice.div_(weights_slice)
            pred_slice = torch.argmax(probs_slice, dim=1).to(torch.uint8).cpu()
            pred_seg[:, :, :, slab_start: slab_start + remaining_len] = pred_slice

    return pred_seg.squeeze()[..., :d_orig]


def _probe_slab_budget(device: torch.device, args, activation_headroom_gb: float = 4.0) -> float:
    """
    Query free VRAM after model load and return a safe slab accumulator budget.

    If args.max_gpu_memory_gb is set, the visible GPU memory is capped to that value —
    useful for simulating the challenge T4 (16 GB) when running on a larger GPU.

    Reserves `activation_headroom_gb` for per-patch activations, forward-pass
    intermediates, and CUDA allocator fragmentation.

    On CPU returns a large sentinel (no VRAM constraint).
    Floors at 1.0 GB so the stripe selector always has a valid value.
    """
    if device.type != 'cuda':
        return 999.0
    # mem_get_info requires an explicit index; torch.device('cuda') (no index) raises ValueError
    device_idx = device.index if device.index is not None else torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
    used_bytes = total_bytes - free_bytes

    max_gb = getattr(args, 'max_gpu_memory_gb', None)
    if max_gb is not None:
        # Cap total to simulate a smaller GPU; adjust free accordingly.
        capped_total_bytes = int(max_gb * 1024**3)
        free_bytes = max(0, capped_total_bytes - used_bytes)
        total_bytes = capped_total_bytes

    free_gb  = free_bytes  / 1024**3
    total_gb = total_bytes / 1024**3
    budget_gb = max(1.0, free_gb - activation_headroom_gb)
    print(
        f"[adaptive VRAM] GPU total={total_gb:.1f} GB  "
        f"free after model load={free_gb:.1f} GB  "
        f"headroom reserved={activation_headroom_gb:.1f} GB  "
        f"→ slab budget={budget_gb:.1f} GB"
        + (f"  (capped to {max_gb:.0f} GB)" if max_gb is not None else "")
    )
    return budget_gb


def run_inference(input_tensor, args, device, transform) -> np.ndarray:
    model = DWNet(spatial_dims=3, in_channels=1, out_channels=args.out_channels, act=args.activation, norm=args.norm,
                    bias=False, backbone_name=args.backbone_name, configuration=args.configuration)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device, weights_only=True)['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    # Resolve the effective slab VRAM budget.
    # If use_adaptive_budget=True, probe free VRAM and take min(probe, config) so the
    # config value still acts as a hard ceiling. If False, use the config value directly.
    if getattr(args, 'use_adaptive_budget', False):
        try:
            adaptive_budget = min(_probe_slab_budget(device, args), args.slab_vram_budget_gb)
            print(f"[adaptive VRAM] probe succeeded, effective budget: {adaptive_budget:.2f} GB")
        except Exception as e:
            adaptive_budget = args.slab_vram_budget_gb
            print(f"[adaptive VRAM] probe failed ({e}), falling back to config: {adaptive_budget:.2f} GB")
    else:
        adaptive_budget = args.slab_vram_budget_gb
        print(f"[adaptive VRAM] probe disabled, using config budget: {adaptive_budget:.2f} GB")
    args = dataclasses.replace(args, slab_vram_budget_gb=adaptive_budget)

    start_time = time.time()

    def _run(current_args):
        with torch.inference_mode(), torch.amp.autocast(enabled=True, dtype=torch.float16, device_type=device.type):
            return memory_efficient_inference_final(
                current_args, model, input_tensor["image"], current_args.patch_size,
                device=device, device_accum='gpu', overlap=0.1, blend_mode="gaussian",
                sigma_scale=0.125, cast_dtype=torch.float16, dist_map_cast_dtype=torch.float16,
                is_memfile=False,
            )

    oom_retry = getattr(args, 'oom_retry', True)
    current_args = args
    multiclass_pred = None
    for attempt in range(3):  # max 3 attempts: budget → budget/2 → budget/4
        try:
            multiclass_pred = _run(current_args)
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            # RuntimeError("CUDA out of memory...") is how OOM surfaces in older PyTorch.
            if isinstance(exc, RuntimeError) and "out of memory" not in str(exc).lower():
                raise  # not an OOM — propagate immediately
            if not oom_retry or attempt == 2:
                print(f"[OOM] CUDA out of memory on attempt {attempt+1}/3, no more retries.")
                raise
            torch.cuda.empty_cache()
            new_budget = current_args.slab_vram_budget_gb / 2.0
            print(f"[OOM retry {attempt+1}] CUDA OOM — halving slab budget "
                  f"{current_args.slab_vram_budget_gb:.1f} → {new_budget:.1f} GB and retrying...")
            current_args = dataclasses.replace(current_args, slab_vram_budget_gb=new_budget)

    if multiclass_pred is None:
        raise RuntimeError("run_inference: all OOM retry attempts failed — multiclass_pred is None")

    model = free_tensor(model)

    print("inference time: %.2f seconds" % (time.time() - start_time))

    multiclass_pred = multiclass_pred.to(dtype=torch.int32, device=device)
    remapped = remap_labels_torch(multiclass_pred, pred_to_challange_map)
    
    prediction = transform.post_inference_transform({"pred": MetaTensor(remapped.unsqueeze(0)), "image": input_tensor["image"][0]})["pred"] # B,H,W,D
    return prediction.squeeze().cpu().numpy().astype(np.int32) # H,W,D