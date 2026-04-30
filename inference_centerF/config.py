from dataclasses import dataclass
from typing import Tuple


@dataclass
class Args:
    # ── Model ─────────────────────────────────────────────────────────────────
    patch_size: Tuple[int, int, int] = (256, 256, 128)
    n_features: int = 32
    unet_depth: int = 5
    norm: str = "instance"
    activation: str = "relu"
    backbone_name: str = "resnet34"
    out_channels: int = 47          # 0=bg, 1-10=anatomy, 11-42=teeth, 43-45=canals, 46=pulp
    configuration: str = "UNET"    # DIST_DIR checkpoint loaded with strict=False

    # ── Preprocessing ─────────────────────────────────────────────────────────
    # centerF was trained at 0.6 mm isotropic — keep this matching the training config.
    pixdim: float = 0.6
    houndsfield_clip: int = 3000    # HU clipped to [0, 3000] then normalised to [0, 1]

    # ── Checkpoint ────────────────────────────────────────────────────────────
    # Path to the stripped .pth file (relative to the inference_centerF folder,
    # or absolute).  Overridable from the CLI: --checkpoint /path/to/file.pth
    checkpoint_path: str = "checkpoints/model.pth"

    # ── Sliding-window inference ────────────────────────────────────────────────
    overlap: float = 0.5          # patch overlap fraction passed to sliding_window_inference
    sw_batch_size: int = 4        # number of patches processed in parallel per forward pass

    # ── Output label space ────────────────────────────────────────────────────
    # True  → remap internal class indices (0-46) to challenge FDI labels
    #         (background=0, teeth=11-48, canals=51-53, pulp=50).
    # False → save raw model output (0-46); faster, no remapping overhead.
    remap_to_challenge_labels: bool = True
