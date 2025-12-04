"""
mx_sampler (MetaXpress)
----------------------
Copy random MetaXpress-style well-site time windows from a plate-reader directory
into an output directory, preserving the original filenames.

Expected filename pattern (case-insensitive):
    t{T}_{WELL}_s{SITE}_w{C}_z{Z}.tif
Example:
    t1_B02_s1_w1_z1.tif

Behavior:
- Randomly pick well-sites (e.g., B02_s1).
- If multiple time points exist, pick a random *start* time and copy
  *n_time_frames* consecutive frames starting from that time.
- Copies the original files as-is (keeps filenames), so downstream analysis
  can use the same pipeline as the full dataset.
- By default copies all channels (w indices) and all z-slices (z indices) for
  each selected time in the window. You can restrict channels via an option.

Dimensions copied: T x C x Z (each as separate files), preserving 2D per-file.
"""
from __future__ import annotations

import os
import re
import glob
import random
import shutil
import csv
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple, Union

# -----------------------------
# Filename parsing and indexing
# -----------------------------

# Regex for names: t1_B02_s1_w1_z1.tif
NAME_RE = re.compile(
    r"^t(?P<T>\d+)_"              # time index at start
    r"(?P<well>[A-Za-z]\d{2})_"   # well like B02
    r"s(?P<site>\d+)_"            # site index
    r"w(?P<C>\d+)_"               # channel index
    r"z(?P<Z>\d+)"                # z index
    r"\.(tif|tiff)$",
    re.IGNORECASE,
)

IndexType = Dict[Tuple[str, int], Dict[int, Dict[int, Dict[int, str]]]]
# Structure: index[(well, site)][T][C][Z] = filepath


def build_index(input_dir: str) -> IndexType:
    """Scan directory and build an index mapping (well, site, T, C, Z) to file paths."""
    index: IndexType = {}
    patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(input_dir, p)))

    for path in files:
        name = os.path.basename(path)
        m = NAME_RE.match(name)
        if not m:
            continue
        well = m.group("well").upper()
        site = int(m.group("site"))
        t = int(m.group("T"))
        c = int(m.group("C"))
        z = int(m.group("Z"))
        key = (well, site)
        index.setdefault(key, {}).setdefault(t, {}).setdefault(c, {})[z] = path

    return index


def list_well_sites(input_dir: str) -> List[str]:
    """Return a list of well-site labels like 'B02_s1' present in the directory."""
    idx = build_index(input_dir)
    labels = [f"{well}_s{site}" for (well, site) in idx.keys()]
    labels.sort()
    return labels

# -----------------------------
# Sampling helpers
# -----------------------------

def _valid_start_positions(ts: List[int], n_time_frames: int) -> List[int]:
    """Return all start t such that a full consecutive window of size n_time_frames exists."""
    if not ts:
        return []
    ts_set = set(ts)
    tmin, tmax = min(ts), max(ts)
    starts: List[int] = []
    for start in range(tmin, tmax - n_time_frames + 2):
        window = list(range(start, start + n_time_frames))
        if all(t in ts_set for t in window):
            starts.append(start)
    return starts


def _gather_window_paths(
    index: IndexType,
    well: str,
    site: int,
    start_t: int,
    n_time_frames: int,
    channels: Optional[Sequence[int]],
) -> Optional[List[str]]:
    """
    Collect all file paths for the given well-site over the requested time window.
    Includes all Z for selected channels (or all channels if None).
    Returns None if any required file is missing.
    """
    site_idx = index.get((well, site), {})
    if not site_idx:
        return None

    ts_window = list(range(start_t, start_t + n_time_frames))

    # Determine channels
    if channels is None:
        ch_set = set()
        for t in ts_window:
            ch_set.update(site_idx.get(t, {}).keys())
        ch_list = sorted(ch_set)
    else:
        ch_list = sorted(set(channels))

    # Z indices (union across window and channels)
    z_set = set()
    for t in ts_window:
        for c in ch_list:
            z_set.update(site_idx.get(t, {}).get(c, {}).keys())
    z_list = sorted(z_set)

    if len(z_list) == 0:
        return None

    paths: List[str] = []
    for t in ts_window:
        for c in ch_list:
            for z in z_list:
                p = site_idx.get(t, {}).get(c, {}).get(z)
                if p is None:
                    return None
                paths.append(p)
    return paths

# -----------------------------
# Core: copy samples
# -----------------------------

def copy_plate_dir_samples(
    input_dir: str,
    output_dir: str,
    n_images: int = 10,
    n_time_frames: int = 3,
    channels: Union[str, Sequence[int]] = "all",
    seed: Optional[int] = None,
    overwrite: bool = False,
    subdir_per_sample: bool = False,
    manifest_path: Optional[str] = None,
) -> List[str]:
    """
    Copy random well-site windows from a directory.
    Optionally write a manifest CSV if `manifest_path` is provided.
    """
    rng = random.Random(seed) if seed is not None else random.Random()

    os.makedirs(output_dir, exist_ok=True)

    index = build_index(input_dir)
    site_keys = list(index.keys())
    if len(site_keys) == 0:
        return []

    # Pick random well-sites without replacement
    k = min(n_images, len(site_keys))
    chosen_sites = rng.sample(site_keys, k)

    copied: List[str] = []

    # Manifest setup
    manifest_rows: List[List[str]] = []
    manifest_header = [
        "sample_id","well","site","t_start","t_end","channels","subdir_per_sample","src","dst","run_timestamp","seed"
    ]

    # Normalize channels parameter
    user_channels: Optional[Sequence[int]]
    if isinstance(channels, str) and channels.lower() == "all":
        user_channels = None
    else:
        user_channels = sorted(set(int(x) for x in channels))  # type: ignore[arg-type]

    for well, site in chosen_sites:
        ts_available = sorted(index[(well, site)].keys())
        starts = _valid_start_positions(ts_available, n_time_frames)
        if not starts:
            # No full window exists; skip this site
            continue
        start_t = rng.choice(starts)
        end_t = start_t + n_time_frames - 1

        window_paths = _gather_window_paths(
            index=index,
            well=well,
            site=site,
            start_t=start_t,
            n_time_frames=n_time_frames,
            channels=user_channels,
        )
        if window_paths is None:
            continue

        # Destination directory for this sample
        dst_dir = (
            os.path.join(output_dir, f"{well}_s{site}_t{start_t}-t{end_t}")
            if subdir_per_sample
            else output_dir
        )
        os.makedirs(dst_dir, exist_ok=True)

        for src in window_paths:
            basename = os.path.basename(src)
            dst = os.path.join(dst_dir, basename)
            if os.path.exists(dst) and not overwrite:
                root, ext = os.path.splitext(basename)
                suf = 1
                while True:
                    candidate = os.path.join(dst_dir, f"{root}_{suf}{ext}")
                    if not os.path.exists(candidate):
                        dst = candidate
                        break
                    suf += 1
            shutil.copy2(src, dst)
            copied.append(dst)
            manifest_rows.append([
                str(chosen_sites.index((well, site)) + 1),
                well, str(site), str(start_t), str(end_t),
                ("all" if user_channels is None else ",".join(map(str, user_channels))),
                str(subdir_per_sample), src, dst,
                datetime.utcnow().isoformat()+"Z", str(seed) if seed is not None else ""
            ])

    # Write manifest if requested
    if manifest_path:
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w", newline="", encoding="utf-8") as mf:
            w = csv.writer(mf)
            w.writerow(manifest_header)
            w.writerows(manifest_rows)

    return copied


# -----------------------------
# CLI
# -----------------------------

import click

@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=str))
@click.option("--output-dir", "output_dir", type=click.Path(file_okay=False, path_type=str), default="plate_samples_copy", help="Directory to copy selected files.")
@click.option("--n-images", "n_images", type=int, default=10, show_default=True, help="Number of well-site samples to export.")
@click.option("--n-time-frames", "n_time_frames", type=int, default=3, show_default=True, help="Number of consecutive time frames per sample.")
@click.option("--channels", "channels", type=str, default="all", show_default=True, help="Channels (w indices): 'all' or comma-separated integers, e.g., '0,1'.")
@click.option("--seed", "seed", type=int, default=None, help="Random seed for reproducibility.")
@click.option("--overwrite/--no-overwrite", "overwrite", default=False, show_default=True, help="Overwrite existing files if they exist.")
@click.option("--subdir-per-sample/--flat", "subdir_per_sample", default=False, show_default=True, help="Copy files into a subdirectory per sample.")
@click.option("--manifest", "manifest", is_flag=True, default=False, help="Write a manifest CSV to OUTPUT_DIR/manifest.csv.")
@click.option("--manifest-path", "manifest_path", type=click.Path(dir_okay=False, path_type=str), default=None, help="Optional explicit path for the manifest CSV.")
def main(input_dir: str, output_dir: str, n_images: int, n_time_frames: int, channels: str, seed: Optional[int], overwrite: bool, subdir_per_sample: bool, manifest: bool, manifest_path: Optional[str]) -> None:
    """CLI entry point for copying random well-site time windows from a directory."""
    # Parse channels option
    ch: Union[str, Sequence[int]]
    if isinstance(channels, str) and channels.lower().strip() == "all":
        ch = "all"
    else:
        try:
            ch = [int(x.strip()) for x in channels.split(",") if x.strip() != ""]
        except Exception:
            raise click.BadOptionUsage("channels", "Invalid channels specification. Use 'all' or comma-separated indices like '0,1'.")

    # Resolve manifest path
    if manifest and not manifest_path:
        manifest_path = os.path.join(output_dir, "manifest.csv")

    copied = copy_plate_dir_samples(
        input_dir=input_dir,
        output_dir=output_dir,
        n_images=n_images,
        n_time_frames=n_time_frames,
        channels=ch,
        seed=seed,
        overwrite=overwrite,
        subdir_per_sample=subdir_per_sample,
        manifest_path=manifest_path,
    )

    click.echo(f"Copied {len(copied)} files to {output_dir}")
    if subdir_per_sample:
        click.echo("(Files organized into subdirectories per sample)")
    if manifest_path:
        click.echo(f"Manifest: {manifest_path}")
    for p in copied[:20]:  # show first 20 paths for brevity
        click.echo(f"- {p}")


if __name__ == "__main__":
    main()
