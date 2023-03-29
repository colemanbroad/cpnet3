## run with `sh rsyncpull.sh`

# cd ~/Desktop/mpi-remote/project-broaddus/
mkdir green-carpet/

# ext="cpnet3/cpnet-out/Fluo-C2DL-Huh7"
# ext="cpnet3/cpnet-out/Fluo-N3DL-DRO"
# ext="cpnet3/cpnet-out/Fluo-C3DH-A549"
# ext="cpnet3/cpnet-out"

# open $ext
# exit

# ext="rawdata/celegans_isbi/Fluo-N3DH-CE/gtpts1_unscaled.pkl"
# rm -rf $ext/train/glance*/*.png

# mkdir -p $(dirname $ext)

# exit

## filters are ordered, i.e.
## a file is tested against each filter in turn until match which then determines include/exclude


# This is Carine's data! Not flywing
src="/fileserver/green-carpet/alex/data3/labeled_data_cellseg/{greyscale,labels}/"
src="/fileserver/green-carpet/alex/data3/labeled_data_cellseg/greyscales/20150215_fig3_sphere_repeat17_slice11.tif"
tgt="green-carpet/"

# So is this...
src="/fileserver/green-carpet/alex/data3/labeled_data_membranes/images/20150215_fig3_sphere_repeat17_slice11-normalized.tif"
tgt="green-carpet/"

# So is this...
src="/fileserver/green-carpet/alex/sd_dataset/raw/"
tgt="green-carpet/"



      # --delete \
      # --exclude "*_/" \
      # --exclude "glance_*/" \
      # --include "*/" \
      # --include "*.png" \
      # --include "*/history.pkl" \
      # --include "*/matching*.pkl" \
      # --exclude "*" \

rsync -raci \
      --include "WT_25deg_160413-37_P14_*_raw.tif" \
  efal:$src $tgt
