"""
czi_sampler (Zeiss .czi)
------------------------
Randomly sample scenes (and time frames) from a Zeiss .czi file and save them as TIFF images.

- Reading image data: bioio.BioImage
- Reading metadata (plate/well/site): czitools (best-effort; optional)
- Export: tifffile
- Usable as an importable module (functions) and as a CLI via click.
"""
from __future__ import annotations

import os
import re
import random
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tifffile
import csv
from datetime import datetime

try:
    from bioio import BioImage
except Exception as e:
    raise ImportError(
        "bioio is required. Install with `pip install bioio`. Original error: %s" % e
    )

try:
    import czitools  # type: ignore
except Exception:
    czitools = None  # type: ignore

@dataclass
class SceneInfo:
    scene_id: Union[int, str]
    well: Optional[str] = None
    site: Optional[Union[int, str]] = None
    description: Optional[str] = None


def _parse_scene_infos_with_czitools(input_czi: str) -> Dict[Union[int, str], SceneInfo]:
    results: Dict[Union[int, str], SceneInfo] = {}
    if czitools is None:
        return results
    try:
        md = None
        if hasattr(czitools, "read_metadata"):
            try:
                md = czitools.read_metadata(input_czi)
            except Exception:
                md = None
        if md is None:
            reader = None
            for attr in ("CziFile", "CZIFile", "Reader", "CZIReader"):
                if hasattr(czitools, attr):
                    try:
                        reader = getattr(czitools, attr)(input_czi)
                        break
                    except Exception:
                        reader = None
            if reader is not None:
                if hasattr(reader, "meta"):
                    md = reader.meta
                elif hasattr(reader, "metadata"):
                    md = reader.metadata
                elif hasattr(reader, "metadata_xml"):
                    md = reader.metadata_xml
        if md is None:
            return results
        md_str = None
        try:
            if isinstance(md, str):
                md_str = md
            else:
                md_str = json.dumps(md)
        except Exception:
            md_str = None
        if md_str:
            well_pattern = re.compile(r"Well(?:ID|Name)?\"?\s*[:=]\s*\"?([A-H]\d{2})", re.IGNORECASE)
            site_pattern = re.compile(r"Site\"?\s*[:=]\s*\"?(\d+)")
            scene_int_pattern = re.compile(r"Scene(?:Index|ID)\"?\s*[:=]\s*\"?(\d+)")
            scene_str_pattern = re.compile(r"Scene\"?\s*[:=]\s*\"?([A-Za-z0-9_-]+)")
            wells = well_pattern.findall(md_str)
            sites = site_pattern.findall(md_str)
            scene_ints = [int(s) for s in scene_int_pattern.findall(md_str)]
            scene_strs = scene_str_pattern.findall(md_str)
            n_candidates = max(len(scene_ints), len(scene_strs), len(wells), len(sites))
            for i in range(n_candidates):
                scene_id: Union[int, str]
                if i < len(scene_ints):
                    scene_id = scene_ints[i]
                elif i < len(scene_strs):
                    scene_id = scene_strs[i]
                else:
                    scene_id = i
                well = wells[i] if i < len(wells) else None
                site = sites[i] if i < len(sites) else None
                results[scene_id] = SceneInfo(scene_id=scene_id, well=well, site=site)
        else:
            try:
                def get(d, *keys):
                    cur = d
                    for k in keys:
                        if isinstance(cur, dict) and k in cur:
                            cur = cur[k]
                        else:
                            return None
                    return cur
                scenes = get(md, "Information", "Image", "S", "Scenes") or []
                for i, sc in enumerate(scenes):
                    well = get(sc, "Well") or get(sc, "WellID")
                    site = get(sc, "Site")
                    results[i] = SceneInfo(scene_id=i, well=well, site=site)
            except Exception:
                pass
    except Exception:
        return {}
    return results


def list_scenes(input_czi: str) -> List[Union[int, str]]:
    img = BioImage(input_czi)
    scenes = []
    if hasattr(img, "scenes") and isinstance(img.scenes, (list, tuple)) and len(img.scenes) > 0:
        scenes = list(img.scenes)
    else:
        if hasattr(img, "set_scene"):
            i = 0
            while True:
                try:
                    img.set_scene(i)
                    scenes.append(i)
                    i += 1
                except Exception:
                    break
    return scenes


def _get_time_length(img: BioImage) -> int:
    for probe in ("T", "t"):
        try:
            if hasattr(img, "dims") and probe in getattr(img, "dims"):
                return int(img.dims[probe])
        except Exception:
            pass
    t_len = 0
    if hasattr(img, "get_image_data"):
        try:
            for ti in range(0, 4096):
                try:
                    _ = img.get_image_data("CZYX", T=ti)
                    t_len += 1
                except Exception:
                    break
        except Exception:
            pass
    if t_len == 0:
        try:
            _ = img.get_image_data("CZYX")
            t_len = 1
        except Exception:
            t_len = 0
    return t_len


def _read_frame(img: BioImage, t_index: Optional[int], channels: Optional[Sequence[int]]) -> np.ndarray:
    kwargs = {}
    if t_index is not None:
        kwargs["T"] = int(t_index)
    if channels is not None:
        kwargs["C"] = list(channels)
    arr = img.get_image_data("CZYX", **kwargs)
    if hasattr(arr, "compute"):
        arr = arr.compute()
    return np.asarray(arr)


def sample_czi_to_tiffs(
    input_czi: str,
    output_dir: str,
    n_images: int = 10,
    n_time_frames: int = 3,
    channels: Union[str, Sequence[int]] = "all",
    seed: Optional[int] = None,
    overwrite: bool = False,
    manifest_path: Optional[str] = None,
) -> List[str]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    scene_map = _parse_scene_infos_with_czitools(input_czi)

    img = BioImage(input_czi)
    scene_ids = list_scenes(input_czi)
    if len(scene_ids) == 0:
        scene_ids = [0]

    k = min(n_images, len(scene_ids))
    chosen_scenes = random.sample(scene_ids, k)

    written_paths: List[str] = []

    # Manifest
    manifest_rows: List[List[str]] = []
    manifest_header = ["scene","well","site","t_indices","channels","out_path","dtype","shape","run_timestamp","seed"]

    for s in chosen_scenes:
        try:
            img.set_scene(s)
            scene_label = str(s)
        except Exception:
            scene_label = str(s)
            if hasattr(img, "scenes") and s in img.scenes:
                try:
                    img.set_scene(img.scenes.index(s))
                except Exception:
                    pass

        t_len = _get_time_length(img)
        if t_len is None or t_len <= 1:
            t_indices = [None]
        else:
            k_t = min(n_time_frames, t_len)
            t_indices = sorted(random.sample(list(range(t_len)), k_t))

        if isinstance(channels, str) and channels.lower() == "all":
            channel_indices = None
        else:
            channel_indices = list(map(int, channels))  # type: ignore[arg-type]

        frames: List[np.ndarray] = []
        for ti in t_indices:
            frame = _read_frame(img, t_index=ti, channels=channel_indices)
            if frame.dtype == np.float64:
                frame = frame.astype(np.float32)
            frames.append(frame)

        stack = frames[0][None, ...] if len(frames) == 1 else np.stack(frames, axis=0)

        info = scene_map.get(s) or scene_map.get(str(s))
        well = (info.well if info else None) or "well"
        site = (info.site if info else None) or "site"
        t_label = ("t" + "-".join(["all" if ti is None else str(ti) for ti in t_indices]))
        base = f"scene-{scene_label}_{well}-{site}_{t_label}.tif"
        out_path = os.path.join(output_dir, base)

        if os.path.exists(out_path) and not overwrite:
            suf = 1
            while True:
                candidate = os.path.join(output_dir, f"scene-{scene_label}_{well}-{site}_{t_label}_{suf}.tif")
                if not os.path.exists(candidate):
                    out_path = candidate
                    break
                suf += 1

        tifffile.imwrite(
            out_path,
            stack,
            photometric="minisblack",
            metadata={"axes": "TCZYX"},
        )
        written_paths.append(out_path)

        ch_label = "all" if channel_indices is None else ",".join(map(str, channel_indices))
        t_label_list = ["" if ti is None else str(ti) for ti in t_indices]
        manifest_rows.append([
            scene_label, well, str(site), ";".join(t_label_list), ch_label, out_path, str(stack.dtype), "x".join(map(str, stack.shape)), datetime.utcnow().isoformat()+"Z", str(seed) if seed is not None else ""
        ])

    if manifest_path:
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w", newline="", encoding="utf-8") as mf:
            w = csv.writer(mf)
            w.writerow(manifest_header)
            w.writerows(manifest_rows)

    return written_paths


# -----------------------------
# CLI
# -----------------------------

import click

@click.command()
@click.argument("input_czi", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("--output-dir", "output_dir", type=click.Path(file_okay=False, path_type=str), default="czi_samples", help="Directory to write sampled TIFFs.")
@click.option("--n-images", "n_images", type=int, default=10, show_default=True, help="Number of samples (scenes) to export.")
@click.option("--n-time-frames", "n_time_frames", type=int, default=3, show_default=True, help="Number of time frames per sample if time-lapse is present.")
@click.option("--channels", "channels", type=str, default="all", show_default=True, help="Channels to include: 'all' or comma-separated indices, e.g., '0,1'.")
@click.option("--seed", "seed", type=int, default=None, help="Random seed for reproducibility.")
@click.option("--overwrite/--no-overwrite", "overwrite", default=False, show_default=True, help="Overwrite existing files.")
@click.option("--manifest", "manifest", is_flag=True, default=False, help="Write a manifest CSV to OUTPUT_DIR/manifest.csv.")
@click.option("--manifest-path", "manifest_path", type=click.Path(dir_okay=False, path_type=str), default=None, help="Optional explicit path for the manifest CSV.")

def main(input_czi: str, output_dir: str, n_images: int, n_time_frames: int, channels: str, seed: Optional[int], overwrite: bool, manifest: bool, manifest_path: Optional[str]) -> None:
    """CLI entry point."""
    ch: Union[str, Sequence[int]]
    if isinstance(channels, str) and channels.lower().strip() == "all":
        ch = "all"
    else:
        try:
            ch = [int(x.strip()) for x in channels.split(",") if x.strip() != ""]
        except Exception:
            raise click.BadOptionUsage("channels", "Invalid channels specification. Use 'all' or comma-separated indices like '0,1'.")

    if manifest and not manifest_path:
        manifest_path = os.path.join(output_dir, "manifest.csv")

    written = sample_czi_to_tiffs(
        input_czi=input_czi,
        output_dir=output_dir,
        n_images=n_images,
        n_time_frames=n_time_frames,
        channels=ch,
        seed=seed,
        overwrite=overwrite,
        manifest_path=manifest_path,
    )
    click.echo(f"Wrote {len(written)} TIFFs to {output_dir}")
    if manifest_path:
        click.echo(f"Manifest: {manifest_path}")
    for p in written:
        click.echo(f"- {p}")


if __name__ == "__main__":
    main()
