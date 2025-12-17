#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CZI sampler (BioIO version)

Modes:
- Default ("napari"): keep requested consecutive time frames (T) in the stack.
  -> writes arrays in TCZYX (same content you validated in napari).
- --cellpose4: export a *single* timepoint per sample and drop T in the output.
  -> writes arrays as (H,W), (H,W,C), or (Z,H,W,C), channels-last.

Dependencies:
    pip install bioio bioio-czi tifffile

References:
- BioIO overview & API (TCZYX, scenes, get_image_data): https://bioio-devs.github.io/bioio/OVERVIEW.html
- BioImage class (dims, scenes, get_image_data):       https://bioio-devs.github.io/bioio/bioio.BioImage.html
- bioio-czi plugin details:                             https://pypi.org/project/bioio-czi/
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from bioio import BioImage  # BioIO TCZYX interface, scenes, get_image_data
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "This command requires 'bioio' and 'bioio-czi'.\n"
        "Install with: pip install bioio bioio-czi"
    ) from e

try:
    import tifffile
except Exception as e:
    raise RuntimeError("tifffile is required to write TIFF outputs. Please `pip install tifffile`.") from e


# ---------------------------- data classes & utils ----------------------------

@dataclass
class SampleRow:
    scene: Union[int, str]
    well: str
    site: str
    t_indices: str
    t_pick: Optional[int]
    channels: str
    out_path: str
    dtype: str
    shape: str
    mode: str       # "napari" | "cellpose4"
    run_timestamp: str
    seed: int


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _parse_channels(ch: str, n_channels: int) -> List[int]:
    if isinstance(ch, str) and ch.lower() == "all":
        return list(range(n_channels))
    indices = [int(x.strip()) for x in str(ch).split(",") if x.strip() != ""]
    for i in indices:
        if i < 0 or i >= n_channels:
            raise ValueError(f"Channel index {i} is out of bounds for C={n_channels}")
    return indices


# ---------------------------- public API --------------------------------------

def list_scenes(input_czi: Union[str, Path]) -> List[str]:
    """
    Return scene identifiers in a CZI.

    BioIO exposes scene handling via BioImage.scenes and .set_scene().
    """
    img = BioImage(str(input_czi))
    return list(img.scenes)  # e.g., ["Image:0", "Image:1", ...]


def sample_czi_to_tiffs(
    input_czi: Union[str, Path],
    output_dir: Union[str, Path],
    n_images: int = 10,
    n_time_frames: int = 3,
    channels: Union[str, Sequence[int]] = "all",
    seed: Optional[int] = None,
    overwrite: bool = False,
    manifest_path: Optional[Union[str, Path]] = None,
    cellpose4: bool = False,
) -> List[str]:
    """
    Sample scenes and time windows from a CZI and write TIFFs.
    Returns list of output file paths.

    - Default (napari): keep T window (TCZYX).
    - cellpose4=True: pick a single timepoint from that window and drop T in output;
                      write (H,W), (H,W,C), or (Z,H,W,C) (channels-last).
    """
    rng = random.Random(seed)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    img = BioImage(str(input_czi))  # 5D TCZYX by default
    scene_names = list(img.scenes)  # list of scene ids, e.g. ["Image:0", ...]
    if not scene_names:
        raise RuntimeError("No scenes found in input CZI.")
    if n_images > len(scene_names):
        n_images = len(scene_names)

    picked_scenes = rng.sample(scene_names, k=n_images)

    # Helper to get dims sizes from the current scene
    def _scene_dims() -> Tuple[int, int, int, int, int]:
        # Return T, C, Z, Y, X in exactly this order
        # Using data.shape and BioIO's TCZYX standard.
        T, C, Z, Y, X = img.shape  # TCZYX
        return T, C, Z, Y, X

    if manifest_path is None:
        manifest_path = outdir / "manifest.csv"

    rows: List[SampleRow] = []
    saved_paths: List[str] = []

    for si, sname in enumerate(picked_scenes):
        img.set_scene(sname)  # change scene
        T, C, Z, Y, X = _scene_dims()

        # Channels to extract
        if isinstance(channels, str):
            chan_idx = _parse_channels(channels, C)
            chan_label = channels
        else:
            chan_idx = list(channels)
            chan_label = ",".join(map(str, chan_idx))
        if not chan_idx:
            raise ValueError("No channels selected.")

        # Build the consecutive time window (indices in [0, T-1])
        if T < 1:
            raise RuntimeError(f"Scene {sname}: time dimension size is {T}.")
        n_tf = min(max(n_time_frames, 1), T)
        t_start_max = max(T - n_tf, 0)
        t_start = rng.randint(0, t_start_max) if T > n_tf else 0
        t_indices = list(range(t_start, t_start + n_tf))

        stem = Path(input_czi).stem
        tag = "cellpose4" if cellpose4 else "napari"
        out_name = f"{stem}_scene{si:03d}_{tag}.tif"
        out_path = str(outdir / out_name)
        if (not overwrite) and os.path.exists(out_path):
            raise FileExistsError(f"{out_path} exists; use --overwrite to replace.")

        if cellpose4:
            # --- CELLPOSE 4 MODE: pick ONE timepoint, drop T in output ---
            # Choose the middle of the sampled window
            t_pick = t_indices[len(t_indices) // 2]

            # Read array as CZYX (single T slice) then convert to channels-last
            # BioIO returns arrays in requested order and supports per-axis selection.  (TCZYX standard)
            # https://bioio-devs.github.io/bioio/OVERVIEW.html
            arr = img.get_image_data("CZYX", T=t_pick, C=chan_idx)  # (C, Z, Y, X) or (C, Y, X) if Z absent

            # Move channels to last
            if arr.ndim == 4:      # C, Z, Y, X
                arr = np.moveaxis(arr, 0, -1)  # -> (Z, Y, X, C)
                zdim = arr.shape[0]
                cdim = arr.shape[-1]

                # Reduce trivial dims to match {(H,W), (H,W,C), (Z,H,W), (Z,H,W,C)}
                if zdim == 1 and cdim == 1:
                    arr = arr[0, :, :, 0]         # -> (H, W)
                    shape_str = f"(H,W) = {arr.shape}"
                elif zdim == 1 and cdim > 1:
                    arr = arr[0, :, :, :]         # -> (H, W, C)
                    shape_str = f"(H,W,C) = {arr.shape}"
                elif zdim > 1 and cdim == 1:
                    arr = arr[:, :, :, 0]         # -> (Z, H, W)
                    shape_str = f"(Z,H,W) = {arr.shape}"
                else:
                    shape_str = f"(Z,H,W,C) = {arr.shape}"

            elif arr.ndim == 3:    # C, Y, X (no Z)
                arr = np.moveaxis(arr, 0, -1)     # -> (Y, X, C)
                if arr.shape[-1] == 1:
                    arr = arr[:, :, 0]            # -> (H, W)
                    shape_str = f"(H,W) = {arr.shape}"
                else:
                    shape_str = f"(H,W,C) = {arr.shape}"

            else:
                # Extremely rare: if already (Y,X) for single-channel single-Z
                shape_str = f"(H,W) = {arr.shape}"

            bigtiff = (arr.size * arr.dtype.itemsize) > (4 * 1024**3)
            tifffile.imwrite(out_path, arr, bigtiff=bigtiff, imagej=False)

            rows.append(SampleRow(
                scene=sname, well="", site="",
                t_indices=",".join(map(str, t_indices)),
                t_pick=t_pick,
                channels=chan_label,
                out_path=out_path,
                dtype=str(arr.dtype),
                shape=shape_str,
                mode="cellpose4",
                run_timestamp=_now_iso(),
                seed=seed if seed is not None else -1,
            ))
            saved_paths.append(out_path)

        else:
            # --- NAPARI MODE: keep T window as TCZYX ---
            arr = img.get_image_data("TCZYX", T=t_indices, C=chan_idx)  # (T, C, Z, Y, X)
            shape_str = f"(T,C,Z,Y,X) = {arr.shape}"

            bigtiff = (arr.size * arr.dtype.itemsize) > (4 * 1024**3)
            tifffile.imwrite(out_path, arr, bigtiff=bigtiff, imagej=False,
                             metadata={"axes": "TCZYX"})

            rows.append(SampleRow(
                scene=sname, well="", site="",
                t_indices=",".join(map(str, t_indices)),
                t_pick=None,
                channels=chan_label,
                out_path=out_path,
                dtype=str(arr.dtype),
                shape=shape_str,
                mode="napari",
                run_timestamp=_now_iso(),
                seed=seed if seed is not None else -1,
            ))
            saved_paths.append(out_path)

    _write_manifest(Path(manifest_path), rows, overwrite=overwrite)
    return saved_paths


def _write_manifest(path: Path, rows: List[SampleRow], overwrite: bool = False):
    if path.exists() and (not overwrite):
        raise FileExistsError(f"{path} exists; pass --overwrite or change --manifest-path.")
    fieldnames = list(asdict(rows[0]).keys()) if rows else [
        "scene","well","site","t_indices","t_pick","channels","out_path","dtype","shape","mode","run_timestamp","seed"
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


# ---------------------------- CLI ---------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("czi-sampler", description="Export sampled scenes/time windows from CZI to TIFFs.")
    p.add_argument("input_czi", help="Path to .czi")
    p.add_argument("-o", "--output-dir", default="./czi_samples", help="Output directory")
    p.add_argument("-n", "--n-images", type=int, default=10, help="How many scenes to sample")
    p.add_argument("-t", "--n-time-frames", type=int, default=3, help="Consecutive time frames to consider per sample")
    p.add_argument("--channels", default="all", help="'all' or comma-separated indices, e.g. 0,1")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--manifest", action="store_true", help="Write manifest.csv next to outputs (default: on)")
    p.add_argument("--manifest-path", default=None, help="Custom manifest path")
    p.add_argument("--cellpose4", action="store_true",
                   help="Write Cellpose‑4 friendly arrays by dropping T and using HWC/ZHWC.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = _build_argparser()
    args = ap.parse_args(argv)

    manifest_path = args.manifest_path
    if manifest_path is None and args.manifest:
        manifest_path = Path(args.output_dir) / "manifest.csv"

    paths = sample_czi_to_tiffs(
        input_czi=args.input_czi,
        output_dir=args.output_dir,
        n_images=args.n_images,
        n_time_frames=args.n_time_frames,
        channels=args.channels,
        seed=args.seed,
        overwrite=args.overwrite,
        manifest_path=manifest_path,
        cellpose4=args.cellpose4,
    )

    print("\n".join(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
