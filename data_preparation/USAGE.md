# Data preparation

Run from the project root after installing `requirements.txt`. Outputs must be new directories. `splits.json` preserves the fixed sample IDs; add `--limit 2` for a preview.

Root, synthetic and self-captured packages already contain images and graph labels. For LVD and MIPDB, see the real examples in the main README and the [paper, Appendix: Dataset Details](https://doi.org/10.1007/s11263-026-02882-4).

## Guyot

Download the official RGB/JSON dataset into `raw/guyot/`, retaining `01-TrainAndValidationSet/` and `02-IndependentTestSet/`. This adapter reads the supplied `ParentID` connections; it does not infer new connections.

```bash
python data_preparation/prepare.py guyot --source raw/guyot --dataset guyot --output data/guyot
python data_preparation/prepare.py augment --source data/guyot/train --factor 100 --seed 10 --output data/guyot/train_aug
python data_preparation/prepare.py augment --source data/guyot/val --factor 32 --seed 10 --output data/guyot/val_aug
python data_preparation/prepare.py guyot --source raw/guyot --dataset mix/guyot --output data/mix/guyot
```

Domain IDs: 1,185 train / 63 val / 255 test. Images are resized to 1/4 with cubic interpolation; mix uses 512×512. Ranges are in the main README.

## Web-sourced plants

The URL package includes graph labels, `sources.csv` and crop/resize information. Download each source once and prepare its images:

```bash
python data_preparation/prepare.py download --csv data/mix/toda_world_resample_384/sources.csv --output raw/world
python data_preparation/prepare.py web --source data/mix/toda_world_resample_384 --images raw/world --output prepared/world
```

Review `prepared/world` before placing it at `data/mix/toda_world_resample_384`. Downloads record failures in `unavailable.json`; changed dimensions stop conversion. Some source URLs may no longer be available.

## Leaf backgrounds

Use an existing prepared leaf split (`rgb/`, `data/`, `ids.json`) and matching whole-leaf masks, with dark foreground and light background. The supplied background package contains URLs only. This example assumes you already have the leaf graph labels; it does not construct them from vein masks.

```bash
python data_preparation/prepare.py download --csv sources/lvd_backgrounds/urls.csv --output raw/backgrounds
python data_preparation/prepare.py background --source prepared/leaf --masks raw/lvd/masks --backgrounds raw/backgrounds --limit 2 --seed 10 --output prepared/leaf_background_example
```

The leaf, mask and existing graph are scaled and translated together. Vein masks are separate from whole-leaf masks. New composites do not recreate the fixed historical 500-image test.

## Fixed splits

For an already prepared pool with images and graphs:

```bash
python data_preparation/prepare.py split --source prepared/leaf --dataset mix/lvd_leaf_src --output data/mix/lvd_leaf_src
```

LVD RGB and vein-input splits have separate ID lists. Every prepared split uses `rgb/*.png`, `data/*.npz` and `ids.json`. NPZ fields are `nodes` (normalized x,y) and `edges` (integer index pairs).

## Mtree

`mtree_parameters.py` only samples trunk/branch parameters in scene units:

```bash
python data_preparation/mtree_parameters.py
```

Use these ranges with Blender 2.83 / Mtree 3.1, a 24mm camera and 1024×1024 rendering. The snippet does not create meshes, graphs or projections. Mtree is test-only.

Original providers' terms apply to images and annotations. URLs and examples do not grant redistribution rights.
