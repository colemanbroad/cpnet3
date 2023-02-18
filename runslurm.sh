## Ordered by dataset size

# cpnetargs='DTcP'
cpnetargs='DTP'
sbatch -J "FC2L_Huh7"   -p gpu --gres gpu:1 -n 1 -t   6:00:00 -c 1 --mem 128000 -o slurm_out/"Fluo-C2DL-Huh7".out     -e slurm_out/"Fluo-C2DL-Huh7".err     --wrap "/bin/time -v python cpnet.py Fluo-C2DL-Huh7 $cpnetargs"
sbatch -J "DC2H_HeLa"   -p gpu --gres gpu:1 -n 1 -t   6:00:00 -c 1 --mem 128000 -o slurm_out/"DIC-C2DH-HeLa".out      -e slurm_out/"DIC-C2DH-HeLa".err      --wrap "/bin/time -v python cpnet.py DIC-C2DH-HeLa $cpnetargs"
sbatch -J "PC2H_U373"   -p gpu --gres gpu:1 -n 1 -t   6:00:00 -c 1 --mem 128000 -o slurm_out/"PhC-C2DH-U373".out      -e slurm_out/"PhC-C2DH-U373".err      --wrap "/bin/time -v python cpnet.py PhC-C2DH-U373 $cpnetargs"
sbatch -J "FN2H_GOWT1"  -p gpu --gres gpu:1 -n 1 -t   6:00:00 -c 1 --mem 128000 -o slurm_out/"Fluo-N2DH-GOWT1".out    -e slurm_out/"Fluo-N2DH-GOWT1".err    --wrap "/bin/time -v python cpnet.py Fluo-N2DH-GOWT1 $cpnetargs"
sbatch -J "FC2L_MSC"    -p gpu --gres gpu:1 -n 1 -t   6:00:00 -c 1 --mem 128000 -o slurm_out/"Fluo-C2DL-MSC".out      -e slurm_out/"Fluo-C2DL-MSC".err      --wrap "/bin/time -v python cpnet.py Fluo-C2DL-MSC $cpnetargs"
sbatch -J "FC3L_MDA231" -p gpu --gres gpu:1 -n 1 -t  24:00:00 -c 1 --mem 128000 -o slurm_out/"Fluo-C3DL-MDA231".out   -e slurm_out/"Fluo-C3DL-MDA231".err   --wrap "/bin/time -v python cpnet.py Fluo-C3DL-MDA231 $cpnetargs"
sbatch -J "FN2H_SIM+"   -p gpu --gres gpu:1 -n 1 -t   6:00:00 -c 1 --mem 128000 -o slurm_out/"Fluo-N2DH-SIM+".out     -e slurm_out/"Fluo-N2DH-SIM+".err     --wrap "/bin/time -v python cpnet.py Fluo-N2DH-SIM+ $cpnetargs"
sbatch -J "PC2L_PSC"    -p gpu --gres gpu:1 -n 1 -t   6:00:00 -c 1 --mem 128000 -o slurm_out/"PhC-C2DL-PSC".out       -e slurm_out/"PhC-C2DL-PSC".err       --wrap "/bin/time -v python cpnet.py PhC-C2DL-PSC $cpnetargs"
sbatch -J "FN2L_HeLa"   -p gpu --gres gpu:1 -n 1 -t   6:00:00 -c 1 --mem 128000 -o slurm_out/"Fluo-N2DL-HeLa".out     -e slurm_out/"Fluo-N2DL-HeLa".err     --wrap "/bin/time -v python cpnet.py Fluo-N2DL-HeLa $cpnetargs"
sbatch -J "FN3H_CHO"    -p gpu --gres gpu:1 -n 1 -t   6:00:00 -c 1 --mem 128000 -o slurm_out/"Fluo-N3DH-CHO".out      -e slurm_out/"Fluo-N3DH-CHO".err      --wrap "/bin/time -v python cpnet.py Fluo-N3DH-CHO $cpnetargs"
sbatch -J "FC3H_A5_"    -p gpu --gres gpu:1 -n 1 -t   6:00:00 -c 1 --mem 128000 -o slurm_out/"Fluo-C3DH-A549".out     -e slurm_out/"Fluo-C3DH-A549".err     --wrap "/bin/time -v python cpnet.py Fluo-C3DH-A549 $cpnetargs"
sbatch -J "FC3H_A5SIM"  -p gpu --gres gpu:1 -n 1 -t   6:00:00 -c 1 --mem 128000 -o slurm_out/"Fluo-C3DH-A549-SIM".out -e slurm_out/"Fluo-C3DH-A549-SIM".err --wrap "/bin/time -v python cpnet.py Fluo-C3DH-A549-SIM $cpnetargs"
sbatch -J "BC2L_MuSC"   -p gpu --gres gpu:1 -n 1 -t   6:00:00 -c 1 --mem 128000 -o slurm_out/"BF-C2DL-MuSC".out       -e slurm_out/"BF-C2DL-MuSC".err       --wrap "/bin/time -v python cpnet.py BF-C2DL-MuSC $cpnetargs"
sbatch -J "BC2L_HSC"    -p gpu --gres gpu:1 -n 1 -t   6:00:00 -c 1 --mem 128000 -o slurm_out/"BF-C2DL-HSC".out        -e slurm_out/"BF-C2DL-HSC".err        --wrap "/bin/time -v python cpnet.py BF-C2DL-HSC $cpnetargs"
sbatch -J "FN3H_SIM+"   -p gpu --gres gpu:1 -n 1 -t   6:00:00 -c 1 --mem 128000 -o slurm_out/"Fluo-N3DH-SIM+".out     -e slurm_out/"Fluo-N3DH-SIM+".err     --wrap "/bin/time -v python cpnet.py Fluo-N3DH-SIM+ $cpnetargs"
sbatch -J "FN3H_CE"     -p gpu --gres gpu:1 -n 1 -t  24:00:00 -c 1 --mem 128000 -o slurm_out/"Fluo-N3DH-CE".out       -e slurm_out/"Fluo-N3DH-CE".err       --wrap "/bin/time -v python cpnet.py Fluo-N3DH-CE $cpnetargs"
sbatch -J "FC3H_H157"   -p gpu --gres gpu:1 -n 1 -t   6:00:00 -c 1 --mem 128000 -o slurm_out/"Fluo-C3DH-H157".out     -e slurm_out/"Fluo-C3DH-H157".err     --wrap "/bin/time -v python cpnet.py Fluo-C3DH-H157 $cpnetargs"
sbatch -J "FN3L_DRO"    -p gpu --gres gpu:1 -n 1 -t   6:00:00 -c 1 --mem 128000 -o slurm_out/"Fluo-N3DL-DRO".out      -e slurm_out/"Fluo-N3DL-DRO".err      --wrap "/bin/time -v python cpnet.py Fluo-N3DL-DRO $cpnetargs"
sbatch -J "FN3L_TRIC"   -p gpu --gres gpu:1 -n 1 -t   6:00:00 -c 1 --mem 128000 -o slurm_out/"Fluo-N3DL-TRIC".out     -e slurm_out/"Fluo-N3DL-TRIC".err     --wrap "/bin/time -v python cpnet.py Fluo-N3DL-TRIC $cpnetargs"
