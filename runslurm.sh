sbatch -J A549      -p gpu --gres gpu:1 -n 1 -t  6:00:00 -c 1 --mem 128000 -o slurm_out/"A549".out      -e slurm_out/"A549".err      --wrap '/bin/time -v python cpnet.py Fluo-C3DH-A549'
sbatch -J A549-SIM  -p gpu --gres gpu:1 -n 1 -t  6:00:00 -c 1 --mem 128000 -o slurm_out/"A549-SIM".out  -e slurm_out/"A549-SIM".err  --wrap '/bin/time -v python cpnet.py Fluo-C3DH-A549-SIM'
sbatch -J C2DH-HeLa -p gpu --gres gpu:1 -n 1 -t  6:00:00 -c 1 --mem 128000 -o slurm_out/"C2DH-HeLa".out -e slurm_out/"C2DH-HeLa".err --wrap '/bin/time -v python cpnet.py DIC-C2DH-HeLa'
sbatch -J CE        -p gpu --gres gpu:1 -n 1 -t  6:00:00 -c 1 --mem 128000 -o slurm_out/"CE".out        -e slurm_out/"CE".err        --wrap '/bin/time -v python cpnet.py Fluo-N3DH-CE'
sbatch -J CHO       -p gpu --gres gpu:1 -n 1 -t  6:00:00 -c 1 --mem 128000 -o slurm_out/"CHO".out       -e slurm_out/"CHO".err       --wrap '/bin/time -v python cpnet.py Fluo-N3DH-CHO'
sbatch -J DRO       -p gpu --gres gpu:1 -n 1 -t  6:00:00 -c 1 --mem 128000 -o slurm_out/"DRO".out       -e slurm_out/"DRO".err       --wrap '/bin/time -v python cpnet.py Fluo-N3DL-DRO'
sbatch -J GOWT1     -p gpu --gres gpu:1 -n 1 -t  6:00:00 -c 1 --mem 128000 -o slurm_out/"GOWT1".out     -e slurm_out/"GOWT1".err     --wrap '/bin/time -v python cpnet.py Fluo-N2DH-GOWT1'
sbatch -J H157      -p gpu --gres gpu:1 -n 1 -t  6:00:00 -c 1 --mem 128000 -o slurm_out/"H157".out      -e slurm_out/"H157".err      --wrap '/bin/time -v python cpnet.py Fluo-C3DH-H157'
sbatch -J HSC       -p gpu --gres gpu:1 -n 1 -t  6:00:00 -c 1 --mem 128000 -o slurm_out/"HSC".out       -e slurm_out/"HSC".err       --wrap '/bin/time -v python cpnet.py BF-C2DL-HSC'
sbatch -J Huh7      -p gpu --gres gpu:1 -n 1 -t  6:00:00 -c 1 --mem 128000 -o slurm_out/"Huh7".out      -e slurm_out/"Huh7".err      --wrap '/bin/time -v python cpnet.py Fluo-C2DL-Huh7'
sbatch -J MDA231    -p gpu --gres gpu:1 -n 1 -t  6:00:00 -c 1 --mem 128000 -o slurm_out/"MDA231".out    -e slurm_out/"MDA231".err    --wrap '/bin/time -v python cpnet.py Fluo-C3DL-MDA231'
sbatch -J MSC       -p gpu --gres gpu:1 -n 1 -t  6:00:00 -c 1 --mem 128000 -o slurm_out/"MSC".out       -e slurm_out/"MSC".err       --wrap '/bin/time -v python cpnet.py Fluo-C2DL-MSC'
sbatch -J MuSC      -p gpu --gres gpu:1 -n 1 -t  6:00:00 -c 1 --mem 128000 -o slurm_out/"MuSC".out      -e slurm_out/"MuSC".err      --wrap '/bin/time -v python cpnet.py BF-C2DL-MuSC'
sbatch -J N2DH-SIM+ -p gpu --gres gpu:1 -n 1 -t  6:00:00 -c 1 --mem 128000 -o slurm_out/"N2DH-SIM+".out -e slurm_out/"N2DH-SIM+".err --wrap '/bin/time -v python cpnet.py Fluo-N2DH-SIM+'
sbatch -J N2DL-HeLa -p gpu --gres gpu:1 -n 1 -t  6:00:00 -c 1 --mem 128000 -o slurm_out/"N2DL-HeLa".out -e slurm_out/"N2DL-HeLa".err --wrap '/bin/time -v python cpnet.py Fluo-N2DL-HeLa'
sbatch -J N3DH-SIM+ -p gpu --gres gpu:1 -n 1 -t  6:00:00 -c 1 --mem 128000 -o slurm_out/"N3DH-SIM+".out -e slurm_out/"N3DH-SIM+".err --wrap '/bin/time -v python cpnet.py Fluo-N3DH-SIM+'
sbatch -J PSC       -p gpu --gres gpu:1 -n 1 -t  6:00:00 -c 1 --mem 128000 -o slurm_out/"PSC".out       -e slurm_out/"PSC".err       --wrap '/bin/time -v python cpnet.py PhC-C2DL-PSC'
sbatch -J TRIC      -p gpu --gres gpu:1 -n 1 -t  6:00:00 -c 1 --mem 128000 -o slurm_out/"TRIC".out      -e slurm_out/"TRIC".err      --wrap '/bin/time -v python cpnet.py Fluo-N3DL-TRIC'
sbatch -J U373      -p gpu --gres gpu:1 -n 1 -t  6:00:00 -c 1 --mem 128000 -o slurm_out/"U373".out      -e slurm_out/"U373".err      --wrap '/bin/time -v python cpnet.py PhC-C2DH-U373'

# Actual Runtime
# A549        0:04.70
# A549-SIM    0:04.71
# C2DH-HeLa   3:29.91
# CE          3:18:39
# CHO         10:50.10
# DRO         -- need to implement SPARSE
# GOWT1       7:17.23
# H157        -- too much data. must downscale more.
# HSC         0:03.98
# Huh7        0:13.04
# MDA231      3:18:39
# MSC         4:58.38
# MuSC        0:04.19
# N2DH-SIM+   3:34.45
# N2DL-HeLa   5:12.21
# N3DH-SIM+   0:07.62
# PSC         3:41.34
# TRIC        0:09.34
# U373        3:42.05
# TRIF        -- no data yet