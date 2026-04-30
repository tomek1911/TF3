from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Args:
    patch_size: Tuple[int, int, int] = (256, 256, 128)
    pixdim: float = 0.3
    houndsfield_clip: int = 3000
    key: str = "image"
    n_features: int = 32
    unet_depth: int = 5
    norm: str = "instance"
    activation: str = "relu"
    backbone_name: str = "resnet34"
    out_channels: int = 47  # 0=bg, 1-10=anatomy, 11-42=teeth, 43-45=canals, 46=pulp
    classes: int = 46
    configuration: str = "UNET"  # dist/dir decoders not needed at inference; loaded with strict=False
    inference_autocast_dtype: str = "float16"
    checkpoint_path: str = "checkpoints/disapointed_still_525_model_epoch_340_new.pth"
    # VRAM budget (GB) for the slab accumulator (probs + weights).
    # Auto-selects 1/2/4 stripes so the per-stripe buffer stays under this budget.
    slab_vram_budget_gb: float = 3.0
    # If True, probe free VRAM after model load and use min(probe, slab_vram_budget_gb)
    # as the effective budget. Set to False to always use slab_vram_budget_gb directly
    # (safer / more predictable on the challenge server).
    use_adaptive_budget: bool = False
    # Cap total visible GPU memory to simulate the challenge T4 (16 GB).
    # _probe_slab_budget will treat free VRAM as if the GPU has at most this much total.
    # Set to None to use the real GPU capacity.
    max_gpu_memory_gb: float = 16.0
    # If True, on CUDA OOM the inference retries automatically with doubled stripe count
    # (budget halved) up to 4 stripes. Set to False to disable and let OOM propagate.
    oom_retry: bool = True
    # If True, process.py runs synthetic memory/correctness tests instead of real inference.
    test_mode: bool = False