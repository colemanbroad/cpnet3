# Draw tracking tail on top of GT links

Show how predictions compare with GT visually. For detection we show detected
centerpoints right next to each other so it's obvious if they match. How can
we do the same thing for tracking (with GT) so that the differences are
visually apparent?

# Evaluate tracking on droso wing

Use real GT and compare with accuracy from Corinna's Thesis? Find and use her
data? Use funke epithelial tracking model?

Is training stable on DRO dataset?

# Is patch padding effective?

Evaluate performance from patch padding for context during training. Does
it help to add padding to patches for extra context? Try comparing these
approaches on various datasets. Train on 01 test on 02 and vice versa.

!!! WARNING Large border regions can slow down training significantly,
    especially for 3D data. The training rate is proportional to `|m|/|x|`
    where `m` is the masked region and `x` is the entire input patch.

One alternative is to split images into training/validation/testing but not
patches. Then within images we can sample freely, even at oblique angles.

## Getting patch size right for small images

How can we deal with datasets where the image size is less than
`splitIntoPatches.outer_shape` in some dimension? 1. ~~changing / improving
PyTorch backend.~~ The problem is the input to the net is not a multiple of
`divisor`, ~~which causes `MaxPool` to throw an error.~~ NOPE: It causes
`torch.cat()` to throw an error! The pool succeeds, but it produces an image of
the wrong size, because size is computed with `floor(d/2)` instead of `ceil
(d/2)`... Then later the image is upscaled 2^n so it's a perfect multiple of 2.
So if we use either `floor` or `ceil` we throw away information that is
necessary to get the shape right. Otherwise there is nothing wrong with images
of any strange shape. 1. The pytorch backend allows choosing boundary
conditions for convolutions. Is this the same problem? NO. This is different.
The problem is that `MaxPool` fails on dimensions of odd size, because it's
ambiguous whether the LEFT or RIGHT side should be treaded differently... 2.
change image size 1. ~~padding img to multiple of `8`~~ - This just allows us
to guarantee that all patches are divisible by(1,8,8) even if img_size <
desired patch_size. This fixes our pooling problem. But it means our patches
contain padding which is confusing to look at and may introduce bugs... IT DOES
CAUSE BUGS! Because target creation happens _after_ padding we end up putting
non-zero weight on these background areas... But this can be fixed by adjusting
the mask... But then the masking has to know about the padding... Which means
we have to do them jointly. It might be easier just to vary our rescaling?
NOTE: we could do this right by FIRST making target, THEN padding both raw and
target, then splitting up into patches. 2. ~~WINNER: rescaling img~~ to
multiple of `8` ::: Images in A549 are different sizes! And too small. We have
differently sizes images whenever we mix data from different acquisitions. Thus
we have to pick a rescale size dynamically based on input size. FAIL: rescaling
by small amounts that are not a power of two introduce aliasing! And it's
strong 🥺 2. ~~changing patch size~~ keep patches as big as possible for max
context 1. make it roughly _half_ the awkward image dimension


## Getting patch size right for prediction AND training

Q: why do predictions freak out when they see black padding in Fluo-C3DH-A549?
Q: how can I get rid of border effects on predictions? 1. Make the borders very
broad 2. separate outer patches by a multiple of 8.

Point (2) above is _in addition_ to making the outer-patch-size a multiple of 8,
but is only a concern during prediction. Together, this means we either have to
rescale images to sizes that are multiples of 8, or we have to pad them. During
training, we really want to augment patches by shifting them by a few pixels so
that we don't always have the same maxpool groups. The spac

1. Shrink the net size (decrease param counts) to allow expanding the window? 2.
Don't pad with const zero, but change zoom() so it's scales data to multiples
of 8. 3. Determine img size, zoom, patch size, etc ahead of time to simplify
code. This precludes datasets with variable image sizes. 4. 


# Finish detection and tracking on sparse ISBI data

During validation on incomplete datasets (DRO, TRIC, TRIF) plot precision, not
F1, because we don't know the number of False Negatives in the patch. 

BF-C2DL-HSC and BF-C2DL-MuSC are "sparse", even though they're densely
annotated... But this is spatial sparsity, which is a different kind.

Should we oversample/upweight patches depending on the number of annotations?

Add filtering to predictions on datasets with `ignore FP` based on first GT
frame.

# Compare vs full instance segmentation methods on C. elegans data

Stardist and other methods use this dataset as the best example of full 3D
instance segmentation data for nuclei, but the actual scores aren't that
good. I think my MICCAI 2023 papers have a reference to the downloadable data (should be freely available now).

# Implement More Linking Methods

_Pre and Post processing_
-[ ] Center of mass alignment
-[ ] Local center of mass alignment? (i.e. optical flow?)
-[x] prune singletons
-[ ] fills tracking gaps

-[x] Nearest Euclidean neib
-[x] Nearest Euclidean neib with max children constraint
-[x] Optimal assignment between frames with max children constraint
    -[ ] Show that optimal assignment between frame pairs fails where VelGrad succeeds.
    -[ ] How do Munkres methods tune (learn) costs ? Assume Euclidean? How set ENTER/EXIT costs? Learn from data?
    -[ ] Optimal Assignment with SQUARED euclidean distance costs
-[ ] Viterbi ala Magnussen for optimal single cell trajectories
-[ ] Tracking by Assignment
    -[ ] subset of frames / tile space and time
    -[ ] all frames
    -[ ] via segtools.track_tools
    -[ ] directly with scipy.linsolve
-[ ] Tracking by Assignment with velocity gradient costs
    -[ ] Fast VelGrad with lookahead solver
-[ ] Joint registration and assignment on framepairs (with velgrad costs) ala Dagmar & Florian
-[ ] "Tracking the Gaps" Track the shapes that constellations of points make instead of the points themselves.
    - Greedy nearest neib in (nd)^3 for triangles and (nd)^p for p-gons (gaps)
    - no canonical order a,b,c vs acb vs ...
    + so use set of pts and set similarity or use (mean(abc) , area(abc))
    - there ARE situations where it's easier to track these shapes-in-the-gaps than the original points!
-[ ] Joint nonlinear registration and correspondence (assignment) with soft nearest parents.
    We 

# Oversaturated glance patches

Q: Why do the raw parts of the training data patches look oversaturated?
    [see here](file:///Users/broaddus/Desktop/work/cpnet3/inspect_data/Fluo-N3DH-SIM+-t095-d1600.png)
A: Bug in png creation


# Simulate smooth warp fields to test velgrad tracker

# Why is cluster disk throughput slow?

Writing to disk (throughput! not latency) on cluster is about 6x slower than
on laptop. I'm getting write speeds (writing pngs from python) of 7.34
MiB/s. 


# Move dataset specific params to CSV

All isbi dataset dependent params should be specified in CSV or some other
kind of easily visible and hand-editable database. CSV editors make this much
easier than dealing with the same kind of data in a text editor. The language
and syntax for data literals don't matter! None of them are as good as a
dedicated spreadsheet editor.

# Why isn't Fluo-N3DH-CE accuracy at 0.99 ?

Weren't the scores in the previous thesis version of cpnet 0.99 on C. elegans
for `train=01 test=02` ? And vice versa? Or was that only for `traintimes=
[3,80,150] predtimes=remaing`? 

# Implement augmentations from MICCAI submission


see ![augmentations](augmentations.png)

> We randomly crop 643 subvolumes, followed by affine spatial deformations
  (translations, rotations, scales, and shears) with reflection padding used
  to simulate variable instance densities. We then employ several intensity
  augmentations including (in order) random bias fields, k-space spikes,
  gamma adjustments, Gaussian blurring along each axis independently to
  simulate partial voluming, Gibbs ringing, sharpening, and cutout. This is
  followed by spatial deformations using random axis- aligned flips, 90◦
  rotations, and elastic deformations with zero padding. Zero padding and
  cutout are used to simulate blank regions common in bioimaging. Finally, we
  add Gaussian, Poisson, or speckle noise to the non-zero regions of 20% of
  the images.

> We therefore build on the shapes generative model of [13] to simulate b ∼ U
  {1,B} random geo- metric shapes in the background

# Prediction time augmentation

I suspect that rotating my samples at prediction time will help improve
`DIC-C2DH-HeLa` which has terrible annotations and often predicts in the
wrong spot altogether. 

# Allow cells to enter and exit through boundaries

This is an extension of allowing cells to enter and exit AT ALL. You could restrict ENTER/EXIT to a boundary zone near image edge? This would be useful. 

# Why does NN tracking fail on BF-C2DL-MuSC ?

Does it fail? The evidence for failure is the relatively low score of the NN
linker compared to other datasets. How can I determine why? You want to see the links that are made and correlate errors with strange behavior in dataset.

This is purely a linking issue, so the data lives in `track-analysis/`.
I think the initial experiments were run remotely... 

Actually, there are five datasets that don't do well with NN linking on GT pts.

Fluo-C2DL-MSC       02    0.934911
BF-C2DL-MuSC        01    0.956835
Fluo-C3DH-H157      02    0.962199
Fluo-C3DL-MDA231    02    0.9746
BF-C2DL-MuSC        02    0.975593

Let's start with the worst one, Fluo-C2DL-MSC/02.

- A small number of cells so any errors are costly.
- The cells are constantly appearing and disappearing! It's actually amazing
  that we get 0.93 F1 at all, given that we guarantee a mistake whenever a cell enters the FoV through an image boundary.
- The 01 acquisition has more cells and they don't enter through the boundaries as often.

Potential solution: add a distance cutoff to the edges and allow for edges without parents.

Now BF-C2DL-MuSC/01.

The cells crawl around in a dish, so no cells enter or leave the FoV through the image boundaries. But they do divide A LOT towards the end of the series. It goes from 1 to 24 cells from t=911 to t=1376. If we plot the F1 score over time I bet all the errors are at the end. 

-[x] ![write code to make this plot](f1-over-time.png). 

The banding in the plot show that we have discrete numbers of errors starting with zero. It's true! The F1 score for this dataset drops dramatically towards the end of the timeseries and fluctuates wildly. It would be nice to know _exactly where_ the problems arose. What kind of movements / divisions / etc cause these problems. see #23. 

_These scores are an indictment of the nearest parent strategy?_

What should we do about this? Let's look at the linking stats by category.

```
YP (row)    GT (col)

move , divide , miss
4937 , 0      , 25
195  , 44     , 208
234  , 0      , 0
```

So GT divisions always get correctly identified as divisions, the problem lies in the following:

- GT movements that get misclassified as divisions (195) 
- GT movements that are missed completely (234)
- YP movements that don't exist (25)
- YP divisions that don't exist (208) !!! This is half of all divisions!

Remember that if we assign to the wrong parent we create TWO yp divisions that are false: one in the GT miss category and one in the GT move category. So of the 208 YP divisions->miss 195 of them are miss a GT movement we are creating a false division with one edge that doesn't exist AND we're missing a GT edge that does exist. 

Solution: We need a better model of cell movements.
    - disallow cell death (minimize same cost s.t. this constraint)
    - preference for movement over division in cost

OK, now let's look at Fluo-C3DH-H157/02.

It's the same situation as Fluo-C2DL-MSC/02. The cells are constantly moving in and out of bounds. We should be able to detect these situations without looking at images, but just by looking at GT links. If there are many ENTER and EXIT nodes in the GT lineage then we don't expect to do well.

What about Fluo-C3DL-MDA231/02 ? 

There is no obvious visual cause, although there are many cells near the boundary. The cells have oblong appearance, move quickly and some do enter/exit through image boundaries. The scores are not unexpected given the image. Still it would be nice to know what fraction of these scores come from enter/exits vs bad movement model...

Collecting the solutions above we have:
-[ ] Tracking model that allows enter and exit through bounds #34
-[x] Know how many enter/exit events we have over time for any dataset from GT anno.

# How many enter/exits are there for each GT dataset?

Let's build a table that computes the stats for every GT dataset. We want to cache (pickle) the tracking so we don't have to load, and we want to process it and compute Divisions, Entries, Exits and the total number of objects over all times. Note, that a single object can count for both an Entry and an Exit at once, or an Entry and a Division.


# Hyperparameter Tuning for Detection and Tracking

# Dynamically adjust batch size to optimize training throughput

# Make a C lib or static linked lib that offers tracking methods that compete head-to-head with TrackMate







