## run with `sh rsyncpull.sh`

cd ~/Desktop/mpi-remote/project-broaddus/

### WARNING: a
# ext="devseg_2/expr/e23_mauricio_jit/v02/"
# ext="rawdata/isbi_challenge_out_extra/trackingvideo/"
# ext="rawdata/isbi_challenge_out_extra/Fluo-N3DL-TRIF/"

ext="cpnet3/data/Fluo-N2DH-GOWT1/"

# open $ext
# exit

# ext="rawdata/celegans_isbi/Fluo-N3DH-CE/gtpts1_unscaled.pkl"
# rm -rf $ext/train/glance*/*.png
mkdir -p $(dirname $ext)

# exit

## filters are ordered, i.e.
## a file is tested against each filter in turn until match which then determines include/exclude

rsync -r \
      --exclude "*_/" \
      --include "*/" \
      --include "*/history.pkl" \
      --include "*.png" \
      --exclude "*" \
  efal:/projects/project-broaddus/$ext $ext
