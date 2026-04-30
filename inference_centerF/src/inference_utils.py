import torch

# ── Label mappings ──────────────────────────────────────────────────────────
# Internal model output: class indices 0-46
#   0        = background
#   1-10     = anatomical structures
#   11-42    = teeth (contiguous; original FDI quad-notation remapped)
#   43-45    = nerve canals
#   46       = merged pulp (centerF training sentinel 100 → class 46)

# Challenge / FDI output label space (remap_to_challenge_labels=True):
#   0        = background
#   1-10     = anatomy
#   11-18, 21-28, 31-38, 41-48  = teeth (original FDI notation)
#   51-53    = nerve canals
#   50       = pulp
pred_to_challenge_map = {
    0: 0,
    1: 1,   2: 2,   3: 3,   4: 4,   5: 5,
    6: 6,   7: 7,   8: 8,   9: 9,   10: 10,
    # teeth quad 1: internal 11-18 → FDI 11-18
    11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18,
    # teeth quad 2: internal 19-26 → FDI 21-28
    19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 26, 25: 27, 26: 28,
    # teeth quad 3: internal 27-34 → FDI 31-38
    27: 31, 28: 32, 29: 33, 30: 34, 31: 35, 32: 36, 33: 37, 34: 38,
    # teeth quad 4: internal 35-42 → FDI 41-48
    35: 41, 36: 42, 37: 43, 38: 44, 39: 45, 40: 46, 41: 47, 42: 48,
    # canals
    43: 51, 44: 52, 45: 53,
    # pulp (centerF class 46 → challenge label 50)
    46: 50,
}


def remap_labels_torch(seg_pred: torch.Tensor, mapping_dict: dict) -> torch.Tensor:
    """
    Remap segmentation labels using a LUT on GPU/CPU.

    Args:
        seg_pred: integer tensor of any shape (model output class indices).
        mapping_dict: {source_class: target_label}.

    Returns:
        Remapped tensor with same shape and dtype int32.
    """
    max_key = max(mapping_dict.keys())
    lut = torch.full((max_key + 1,), -1, dtype=torch.int32, device=seg_pred.device)
    for k, v in mapping_dict.items():
        lut[k] = v
    return lut[seg_pred]
