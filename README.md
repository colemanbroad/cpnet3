**Detect and track cells in bioimage timeseries.**  
Produce results in a variety of formats including [ISBI CTC](http://celltrackingchallenge.net/) compatible.

This project is a consolidation of [devseg_2]
(https://github.com/mpicbg-csbd/devseg2) which reduces all code necessary for
data generation, detection training, detection prediction, segmentation,
tracking and evaluation to 1k lines of python.

    -------------------------------------------------------------------------------
    Language                     files          blank        comment           code
    -------------------------------------------------------------------------------
    Python                           6            324            315           1025
    Markdown                         1             11              0             75
    Bourne Shell                     3             11             39             32
    Text                             1              2              0             15
    -------------------------------------------------------------------------------
    SUM:                            11            348            354           1147
    -------------------------------------------------------------------------------

# Setup

First, make sure you're python environment is set up: `pip install -r requirements.txt`.
Then download some training datasets from isbi: `sh download-isbi-data.sh`.

# Run

Run with `python cpnet.py Fluo-C2DL-Huh7` (or any ISBI dataset name) assuming a directory structure like:

```
Fluo-C2DL-Huh7/
├── 01
│   ├── t000.tif
│   ├── ...
├── 01_GT
│   ├── SEG
│   │   ├── man_seg000.tif
│   │   ├── ...
│   └── TRA
│       ├── man_track.txt
│       ├── man_track000.tif
│       ├── ...
├── 02
│   ├── t000.tif
│   ├── ...
└── 02_GT
    ├── SEG
    │   ├── man_seg000.tif
    │   ├── ...
    └── TRA
        ├── man_track.txt
        ├── man_track000.tif
        ├── ...
```

which produces...

```
cpnet-out/Fluo-C2DL-Huh7/
├── data
│   ├── dataset.pkl ## patches with `raw`, `target` and `mask`.
│   └── png
│       ├── t000-d0000.png ## time:000 - patch:0000
│       └── ...
├── predict
│   ├── t0000-latest.png ## time:0000 - model:latest
│   └── ...
├── track
│   ├── l000.png ## time:000 colored marker for each object
│   └── ...
└── train
    ├── glance_output_train
    │   ├── a000_000.png ## time:000 patch:000 (in training set)
    │   └── ...
    ├── glance_output_vali
    │   ├── a000_000.png ## time:000 patch:000 (in validation set)
    │   └── ...
    ├── history.pkl ## loss and validation scores over time (epochs)
    ├── labels.pkl  ## train|vali label for each patch in `dataset.pkl`
    └── m
        ├── best_weights_f1.pt  ## best model weights according to f1 metric
        ├── best_weights_height.pt
        ├── best_weights_latest.pt
        └── best_weights_loss.pt
```

