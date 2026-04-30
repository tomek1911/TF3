#!/usr/bin/env python3
"""
RAM + VRAM monitor — run in a separate terminal while Docker test executes.

Usage:
    python3 monitor_mem.py [--interval 0.5] [--out mem_log.csv]

Logs every sample to CSV and prints a live compact table.
Prints peak summary on Ctrl+C or natural exit.
"""

import argparse
import csv
import datetime
import subprocess
import sys
import time

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# ── RAM reader (psutil preferred, /proc/meminfo fallback) ────────────────────

def ram_stats_mb():
    """Return (used_mb, total_mb, available_mb, swap_used_mb, swap_total_mb)."""
    if _HAS_PSUTIL:
        v = psutil.virtual_memory()
        s = psutil.swap_memory()
        return (
            v.used   / 1024**2,
            v.total  / 1024**2,
            v.available / 1024**2,
            s.used   / 1024**2,
            s.total  / 1024**2,
        )
    # fallback: /proc/meminfo
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                info[parts[0].rstrip(":")] = int(parts[1])
    total_kb = info["MemTotal"]
    avail_kb = info.get("MemAvailable", info["MemFree"])
    used_kb  = total_kb - avail_kb
    swap_total_kb = info.get("SwapTotal", 0)
    swap_free_kb  = info.get("SwapFree",  0)
    swap_used_kb  = swap_total_kb - swap_free_kb
    return used_kb / 1024, total_kb / 1024, avail_kb / 1024, swap_used_kb / 1024, swap_total_kb / 1024


# ── nvidia-smi reader ────────────────────────────────────────────────────────

def vram_stats_mb():
    """
    Return list of (gpu_index, used_mb, total_mb) for all GPUs.
    Returns empty list if nvidia-smi is unavailable.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode()
    except Exception:
        return []

    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            idx, used, total = int(parts[0]), float(parts[1]), float(parts[2])
            gpus.append((idx, used, total))
    return gpus


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAM + VRAM monitor")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Sampling interval in seconds (default: 0.5)")
    parser.add_argument("--out", default="mem_log.csv",
                        help="Output CSV file (default: mem_log.csv)")
    args = parser.parse_args()

    # Detect GPU count once
    gpus_init = vram_stats_mb()
    n_gpus = len(gpus_init)

    # CSV header
    gpu_headers = []
    for idx, _, _ in gpus_init:
        gpu_headers += [f"gpu{idx}_used_mb", f"gpu{idx}_total_mb"]

    fieldnames = ["timestamp", "ram_used_mb", "ram_total_mb", "ram_avail_mb",
                   "swap_used_mb", "swap_total_mb"] + gpu_headers

    csv_file = open(args.out, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()

    # Peak tracking
    peak_ram_mb  = 0.0
    peak_swap_mb = 0.0
    peak_vram_mb = {idx: 0.0 for idx, _, _ in gpus_init}

    src = "psutil" if _HAS_PSUTIL else "/proc/meminfo"
    print(f"\nRAM source : {src}")
    print(f"Logging to : {args.out}   interval: {args.interval}s   Ctrl+C to stop\n")
    header = (f"{'Time':>8}  {'RAM used':>10} / {'total':>8}  {'swap':>8}   "
              + "  ".join(f"GPU{idx} {'VRAM used':>10} / {'total':>8}" for idx, _, _ in gpus_init))
    print(header)
    print("-" * len(header))

    try:
        while True:
            ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            ram_used, ram_total, ram_avail, swap_used, swap_total = ram_stats_mb()
            gpus = vram_stats_mb()

            row = {
                "timestamp":    ts,
                "ram_used_mb":  round(ram_used,   1),
                "ram_total_mb": round(ram_total,  1),
                "ram_avail_mb": round(ram_avail,  1),
                "swap_used_mb": round(swap_used,  1),
                "swap_total_mb":round(swap_total, 1),
            }

            gpu_cols = ""
            for idx, vram_used, vram_total in gpus:
                row[f"gpu{idx}_used_mb"]  = round(vram_used,  1)
                row[f"gpu{idx}_total_mb"] = round(vram_total, 1)
                peak_vram_mb[idx] = max(peak_vram_mb[idx], vram_used)
                gpu_cols += f"  GPU{idx} {vram_used:>8.0f} MB / {vram_total:>6.0f} MB"

            peak_ram_mb  = max(peak_ram_mb,  ram_used)
            peak_swap_mb = max(peak_swap_mb, swap_used)

            writer.writerow(row)
            csv_file.flush()

            live = (f"{ts:>8}  {ram_used:>8.0f} MB / {ram_total:>6.0f} MB"
                    f"  {swap_used:>6.0f} MB"
                    + gpu_cols)
            print(live, end="\r", flush=True)

            time.sleep(args.interval)

    except KeyboardInterrupt:
        pass
    finally:
        csv_file.close()
        print()  # newline after last \r

    # ── Peak summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("PEAK SUMMARY")
    print("=" * 55)
    print(f"  RAM peak used : {peak_ram_mb:>8.0f} MB  ({peak_ram_mb/1024:.2f} GB)")
    if peak_swap_mb > 0:
        print(f"  Swap peak used: {peak_swap_mb:>8.0f} MB  ({peak_swap_mb/1024:.2f} GB)  *** swap active!")
    for idx, _, total in gpus_init:
        p = peak_vram_mb[idx]
        headroom = total - p
        print(f"  GPU{idx} VRAM peak: {p:>8.0f} MB  ({p/1024:.2f} GB)"
              f"  headroom: {headroom:.0f} MB  total: {total:.0f} MB")
    print("=" * 55)
    print(f"\nFull log saved to: {args.out}\n")


if __name__ == "__main__":
    main()
