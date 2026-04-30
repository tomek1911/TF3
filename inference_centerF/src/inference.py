"""
inference.py — MONAI sliding-window inference for centerF CBCT segmentation.

Public API:
    run_inference(image_tensor, args, device, transforms) -> np.ndarray
"""
import time

import numpy as np
import torch
from monai.data.meta_tensor import MetaTensor
from monai.inferers import sliding_window_inference

from .model import DWNet
from .inference_utils import pred_to_challenge_map, remap_labels_torch


def run_inference(
    image_tensor: torch.Tensor,   # (1, 1, H, W, D) fp16 MetaTensor on device
    args,
    device: torch.device,
    transforms,                   # Transforms instance (for postprocess)
) -> np.ndarray:
    """
    Load model, run MONAI sliding-window inference, remap labels, invert preprocessing.

    Args:
        image_tensor: preprocessed volume as produced by transforms.preprocess().
                      Must already have a batch dimension added: (1,1,H,W,D).
        args:         Args dataclass (config.py).
        device:       torch.device.
        transforms:   Transforms instance (provides postprocess()).

    Returns:
        Segmentation array (H_orig, W_orig, D_orig) np.int32 in original image geometry.
        If args.remap_to_challenge_labels is True the values use FDI / challenge label space;
        otherwise internal class indices 0-46 are returned.
    """
    model = DWNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=args.out_channels,
        act=args.activation,
        norm=args.norm,
        bias=False,
        backbone_name=args.backbone_name,
        configuration=args.configuration,
    )
    ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    t0 = time.time()

    def _predictor(x):
        out = model(x)
        # sliding_window_inference expects a single tensor; return only the seg logits
        return out[0]

    with torch.no_grad(), torch.amp.autocast(
        enabled=True, dtype=torch.float16, device_type=device.type
    ):
        seg_logits = sliding_window_inference(
            image_tensor,
            roi_size=args.patch_size,
            sw_batch_size=4,
            predictor=_predictor,
            overlap=0.5,
            sw_device=device,
            device="cpu",
            mode="gaussian",
            sigma_scale=0.125,
            padding_mode="constant",
            cval=0,
            progress=True,
        )

    pred = seg_logits.argmax(dim=1)   # (1, H, W, D) on CPU
    del seg_logits, model
    torch.cuda.empty_cache()

    print(f"Inference time: {time.time() - t0:.1f}s")

    # ── Label remapping ────────────────────────────────────────────────────
    pred_int = pred.to(dtype=torch.int32)
    if getattr(args, "remap_to_challenge_labels", True):
        pred_int = remap_labels_torch(pred_int, pred_to_challenge_map)

    # ── Invert preprocessing (spacing / orientation / pad) ─────────────────
    result = transforms.postprocess(
        {"pred": MetaTensor(pred_int), "image": image_tensor[0]}
    )["pred"]

    return result.squeeze().cpu().numpy().astype(np.int32)
