# Title

- Robust and general cell tracking
- ~~Bioimage methods for tracking cells during development~~
- Tracking cells through centerpoint detection
- Sparse cell centerpoint annotations are sufficient for tracking

# Abstract


# Contributions

- Demonstration that weak centerpoint annotations alone are sufficient for
  competitive cell tracking across a broad spectrum of real data.
- Achieves best total score in ISBI CTC benchmark, and 1st place cell tracking
  score on four individual datasets
- Only two obvious parameters required (includes both training and prediction)
- Only method we are aware of to run on all 20 datasets
- Capable of training on sparse, weak annotations
- A simple change to the common training patch scheme that improves accuracy
- < 1k lines of python

# Intro / Motivation

Bioimage datasets vary widely, as do approaches for tracking cells.
For instance, it is common in 2D images to track cells by a two step segmentation & linking approach.
But in 3D datasets with anisotropic resolution segmentation is eschewed in favor of predefined, simple
shape models like spheres, ellipsoids, or boxes.
Centerpoint detection models have been used in histology to detect nuclei,
and as part of seeded watershed in 2D segmentation.

Methods that rely on segmentation are more difficult to apply in 3D, and on densely packed objects.
Methods that rely on fitting shapes to fluorescent nuclei are limited to simple shapes (usually H2B nuclei),
and only work on fluorescence images.

Recently methods have explored using centerpoint detection + linking for cell tracking
and skipping segmentation and avoiding cell shape model fitting.
In this paper we explore the generality of this approach, by using the same centerpoint detection +
cell linking method to each of the 20 datasets in the ISBI Cell Tracking Challenge.





- Full 3D instance segmentation annotations are difficult to make.
- Segmentation models based on pixelwise classifiers often undersegment
  densely packed cells in low SNR data.
- Cell centerpoint detection models avoid undersegmentation in densely packed
  objects.

The ISBI CTC is designed with the assumption that segmentation is performed 
either prior to or jointly with cell tracking. This is demonstrated in that the 
"Cell Tracking Benchmark" includes both a segmentation score (SEG) and a
 tracking score (TRA) as well as an overall score which is an average of the
 two. However, historically the most successful methods for tracking cells
 during embryogenesis have eschewed segmentation entirely in favor of
 detect-then-link approaches [StarryNite](bao_automated_2006) and [TGMM]
 (amat_fast_2014). This is because 3D segmentations are more to annotate and
 to verify, and there is little to be gained by a pixel-precise description
 of the highly-undersampled z dimension.

# The method

A detect-then-track approach ...

# Results

We evaluated our detect-then-link approach on the ISBI Cell Tracking and Cell
Segmentation Benchmarks as well as in house datasets.

## Official Evaluations

- Table: TRA scores on all ISBI datasets vs other single method top performers
- Plot: DET vs TRA


- DET Scores on 01/02
- TRA Scores on 01/02

## Internal Experiments

- Detection scores (SNNM) for training with / without context borders (score vs size of data)
- Representative predictions on challenge data (randomly select examples without visual optimization)
  - or select examples with DET scores in the 5th, 50th and 95th percentile.
- Results on 2D Drosophila Wing with polygonal cells and membrane signal (instead of histone signal)
  + #TODO track the wing!
- 





