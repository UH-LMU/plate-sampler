
# plate-sampler

Random sampling utilities for plate-reader datasets (**MetaXpress per-frame TIFFs**) and **Zeiss .czi** files.
Create random subsets for tracking/testing while keeping identical file layout (MetaXpress), or export compact stacks from CZI.

## Install (dev)
```bash
pip install -e .
# Add CZI deps (optional)
pip install -e .[czi]
```

## CLI

### MetaXpress (copy-mode)
```bash
# Copy 8 well-sites, 5 consecutive time frames each, and write a manifest
mx-sampler /path/to/dir --output-dir ./sampled --n-images 8 --n-time-frames 5 --manifest

# Limit channels, reproducible, per-sample subdirs, custom manifest path
mx-sampler /path/to/dir -o ./sampled -n 4 -t 3 --channels 0,1 --seed 123 --subdir-per-sample --manifest-path ./sampled/run1_manifest.csv
```

### Zeiss CZI (stack export)
```bash
# Sample 10 scenes, 3 time frames each, all channels, and a manifest
czi-sampler input.czi --output-dir ./czi_samples --n-images 10 --n-time-frames 3 --channels all --manifest
```

## Programmatic use
```python
from plate_sampler import copy_plate_dir_samples, list_well_sites
from plate_sampler.czi_sampler import sample_czi_to_tiffs, list_scenes

# MetaXpress copy-mode
sites = list_well_sites('/path/to/dir')
copied = copy_plate_dir_samples(
    input_dir='/path/to/dir',
    output_dir='./sampled',
    n_images=6,
    n_time_frames=4,
    channels='all',
    seed=42,
    overwrite=False,
    subdir_per_sample=True,
    manifest_path='./sampled/manifest.csv',
)

# CZI stack export
paths = sample_czi_to_tiffs(
    input_czi='input.czi',
    output_dir='./czi_samples',
    n_images=10,
    n_time_frames=3,
    channels='all',
    seed=1,
    overwrite=True,
    manifest_path='./czi_samples/manifest.csv',
)
```

## Filename pattern (MetaXpress)
```
t{T}_{WELL}_s{SITE}_w{C}_z{Z}.tif
```
Examples: `t1_B02_s1_w1_z1.tif`, `t12_A03_s2_w0_z3.tiff`.

## Manifest columns
- MetaXpress: `sample_id,well,site,t_start,t_end,channels,subdir_per_sample,src,dst,run_timestamp,seed`
- CZI: `scene,well,site,t_indices,channels,out_path,dtype,shape,run_timestamp,seed`

## License
MIT
