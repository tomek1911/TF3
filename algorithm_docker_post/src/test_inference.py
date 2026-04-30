"""
Synthetic inference tests for memory and correctness validation.

Run via process.py when Args.test_mode = True, or directly:
    python -m src.test_inference

Each scenario creates a synthetic fp16 input tensor of a specified (H, W, D) shape,
runs memory_efficient_inference_final, and reports:
  - which stripe mode was auto-selected
  - peak GPU VRAM consumed
  - output shape and value range (expected: 0..out_channels-1)
  - wall-clock inference time

No real CBCT data or preprocessing/postprocessing transforms are involved —
this isolates the sliding-window accumulator which is the memory bottleneck.
"""

import copy
import time
import dataclasses

import torch
import numpy as np

from .model import DWNet
from .inference_ram import memory_efficient_inference_final


# ---------------------------------------------------------------------------
# Test scenario definitions
# Each entry: human-readable name, (H, W, D), optional slab_vram_budget_gb
# override, optional cast_dtype override.
#   budget_override: forces a specific stripe count independent of adaptive probe
#   cast_dtype:      None → fp16 (default); torch.float32 → fp32 slab + div_ path
# ---------------------------------------------------------------------------
SCENARIOS = [
    # name,                  (H,    W,    D),   budget_gb,  cast_dtype
    # ── standard fp16 path ──────────────────────────────────────────────────
    ("tiny_single_zgroup",   (256,  256,  100), None,       None),   # D<pz → z-pad guard
    ("small_multi_z",        (300,  300,  200), None,       None),   # several z-groups, 1 stripe
    ("medium",               (450,  450,  300), None,       None),   # medium XY, 1 stripe
    ("large_xy",             (600,  600,  350), None,       None),   # ~3.5 GB slab, 1 stripe
    ("xlarge_2stripes",      (800,  800,  400), None,       None),   # ~7.5 GB → auto 2 stripes
    ("forced_4stripes",      (800,  800,  600), 2.0,        None),   # budget=2 GB → 4 stripes
    # ── extreme scenarios ───────────────────────────────────────────────────
    ("asym_xy",              (1000, 500,  400), None,       None),   # asymmetric H≠W
    ("max_xy",               (900,  900,  400), None,       None),   # ~9.5 GB slab → 4 stripes
    ("deep_z",               (700,  700,  800), None,       None),   # many z-groups, high D
    ("single_z_maxpatch",    (256,  256,  128), None,       None),   # D==pz exactly, 1 z-group
    # ── fp32 div_ path ──────────────────────────────────────────────────────
    ("fp32_slab_div",        (600,  600,  350), None,       torch.float32),  # fp32 accumulator + div_
    ("fp32_slab_div_large",  (800,  800,  600), None,       torch.float32)
]


def _mb(n_bytes: int) -> str:
    return f"{n_bytes / 1024**2:.0f} MB"


def run_inference_tests(args, device: torch.device) -> None:
    """
    Load the model once, then run all scenarios sequentially.
    Prints a summary table to stdout.
    """
    print("\n" + "=" * 70)
    print("INFERENCE MEMORY / CORRECTNESS TESTS")
    print("=" * 70)
    print(f"Device       : {device}")
    print(f"Model        : {args.backbone_name}, out_channels={args.out_channels}")
    print(f"Patch size   : {args.patch_size}")
    print(f"Default VRAM budget: {args.slab_vram_budget_gb:.1f} GB")
    print("=" * 70)

    # ---- Load model once ------------------------------------------------
    print("Loading model...")
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
    state = torch.load(
        args.checkpoint_path, map_location=device, weights_only=True
    )["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    if device.type == "cuda":
        model_vram = torch.cuda.memory_allocated(device)
        print(f"Model VRAM   : {_mb(model_vram)}")

    results = []

    for name, (H, W, D), budget_override, cast_dtype_override in SCENARIOS:
        print(f"\n[{name}]  shape=({H},{W},{D})", end="")
        if budget_override is not None:
            print(f"  budget_override={budget_override:.1f} GB", end="")
        if cast_dtype_override is not None:
            print(f"  cast_dtype={cast_dtype_override}", end="")
        print()

        # Build a modified args copy if budget is overridden
        test_args = copy.copy(args)
        if budget_override is not None:
            test_args = dataclasses.replace(test_args, slab_vram_budget_gb=budget_override)

        run_cast_dtype = cast_dtype_override if cast_dtype_override is not None else torch.float16

        # Synthetic fp16 input: uniform 0.5 (plausible normalised HU after clipping)
        x = torch.full((1, 1, H, W, D), 0.5, dtype=torch.float16, device=device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            mem_before = torch.cuda.memory_allocated(device)

        t0 = time.time()
        try:
            with torch.inference_mode(), torch.amp.autocast(
                enabled=True, dtype=torch.float16, device_type=device.type
            ):
                pred = memory_efficient_inference_final(
                    test_args,
                    model,
                    x,
                    test_args.patch_size,
                    device=device,
                    device_accum="gpu",
                    overlap=0.1,
                    blend_mode="gaussian",
                    sigma_scale=0.125,
                    cast_dtype=run_cast_dtype,
                    dist_map_cast_dtype=torch.float16,
                    is_memfile=False,
                )
            elapsed = time.time() - t0
            error = None
        except Exception as exc:
            elapsed = time.time() - t0
            pred = None
            error = exc

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_vram = torch.cuda.max_memory_allocated(device)
            delta_vram = peak_vram - mem_before
        else:
            peak_vram = delta_vram = 0

        vram_limit_mb = (
            int(test_args.max_gpu_memory_gb * 1024)
            if device.type == "cuda" and getattr(test_args, "max_gpu_memory_gb", None) is not None
            else None
        )
        vram_exceeded = vram_limit_mb is not None and peak_vram // 2**20 > vram_limit_mb

        if error is not None:
            status = f"ERROR: {error}"
            out_shape = out_min = out_max = "N/A"
        elif vram_exceeded:
            status = f"FAIL(VRAM>{vram_limit_mb}MB)"
            out_shape = tuple(pred.shape)
            out_min   = int(pred.min().item())
            out_max   = int(pred.max().item())
        else:
            status = "OK"
            out_shape = tuple(pred.shape)
            out_min   = int(pred.min().item())
            out_max   = int(pred.max().item())

        print(f"  Status     : {status}")
        print(f"  Output     : shape={out_shape}  range=[{out_min}, {out_max}]  "
              f"(expected [0, {args.out_channels - 1}])")
        print(f"  Time       : {elapsed:.1f}s")
        if device.type == "cuda":
            limit_str = f"  limit={vram_limit_mb} MB" if vram_limit_mb is not None else ""
            print(f"  Peak VRAM  : {_mb(peak_vram)}  (delta from before: {_mb(delta_vram)}){limit_str}")

        results.append({
            "name": name, "shape": (H, W, D),
            "status": status, "time_s": elapsed,
            "peak_vram_mb": peak_vram // 2**20,
            "delta_vram_mb": delta_vram // 2**20,
            "out_min": out_min, "out_max": out_max,
            "cast_dtype": "fp32" if cast_dtype_override == torch.float32 else "fp16",
        })

        # Free the prediction and clear cache between scenarios
        del pred, x
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- Summary table --------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY")
    print(f"{'Scenario':<22} {'Shape':<18} {'dtype':<6} {'Status':<8} {'Time':>6} {'PeakVRAM':>10} {'Range'}")
    print("-" * 80)
    any_fail = any(r["status"] not in ("OK",) for r in results)
    for r in results:
        shape_str = "x".join(str(s) for s in r["shape"])
        range_str = f"[{r['out_min']},{r['out_max']}]"
        flag = " <--" if r["status"] not in ("OK",) else ""
        print(f"  {r['name']:<20} {shape_str:<18} {r['cast_dtype']:<6} {r['status']:<8} "
              f"{r['time_s']:>5.1f}s {r['peak_vram_mb']:>8} MB  {range_str}{flag}")
    if any_fail:
        print("\n*** SOME TESTS FAILED — see rows marked <-- above ***")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    from .config import Args
    _args = Args()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_inference_tests(_args, _device)
