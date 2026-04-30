"""
Strip optimizer, scheduler, and unused decoder weights from a training checkpoint.
Keeps only the keys needed for inference with configuration="UNET":
  backbone_layers, decoder_blocks, segmentation_decoder, multiclass_decoder
Drops: dist_decoder, direction_decoder, pulp_decoder, optimizer, scheduler, metrics.

Usage:
    python strip_checkpoint.py <input.pth> <output.pth>

The output checkpoint has the form:
    {'model_state_dict': <stripped state dict>}
which is compatible with:
    model.load_state_dict(...['model_state_dict'], strict=False)
"""

import argparse
import torch

KEEP_PREFIXES = (
    "backbone_layers.",
    "decoder_blocks.",
    "segmentation_decoder.",
    "multiclass_decoder.",
)

DROP_PREFIXES = (
    "dist_decoder.",
    "direction_decoder.",
    "pulp_decoder.",
)


def strip(input_path: str, output_path: str) -> None:
    ckpt = torch.load(input_path, map_location="cpu", weights_only=True)

    if "model_state_dict" not in ckpt:
        raise KeyError(f"'model_state_dict' not found in checkpoint. Keys: {list(ckpt.keys())}")

    full_state = ckpt["model_state_dict"]
    total = len(full_state)

    stripped_state = {
        k: v for k, v in full_state.items()
        if k.startswith(KEEP_PREFIXES) and not k.startswith(DROP_PREFIXES)
    }

    dropped = total - len(stripped_state)

    out = {"model_state_dict": stripped_state}
    torch.save(out, output_path)

    # Report
    original_mb = sum(v.numel() * v.element_size() for v in full_state.values()) / 1e6
    stripped_mb = sum(v.numel() * v.element_size() for v in stripped_state.values()) / 1e6

    import os
    orig_file_mb = os.path.getsize(input_path) / 1e6
    stripped_file_mb = os.path.getsize(output_path) / 1e6

    print(f"Input:   {total} keys, {original_mb:.1f} MB tensor data, {orig_file_mb:.1f} MB on disk")
    print(f"Output:  {len(stripped_state)} keys, {stripped_mb:.1f} MB tensor data, {stripped_file_mb:.1f} MB on disk")
    print(f"Dropped: {dropped} keys (optimizer, scheduler, dist/dir/pulp decoders, metrics)")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strip training-only data from a DWNet checkpoint.")
    parser.add_argument("input", help="Path to the training checkpoint (.pth)")
    parser.add_argument("output", help="Path for the stripped output checkpoint (.pth)")
    args = parser.parse_args()
    strip(args.input, args.output)
