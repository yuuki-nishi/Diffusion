#! /bin/sh
#$ -cwd
#$ -l s_gpu=1
#it was 24:00:00
#$ -l h_rt=24:00:00
#$ -N calcdiv
#$ -o TsubameOutputs/calcdiv/tmp_o
#$ -e TsubameOutputs/calcdiv/tmp_e
. /etc/profile.d/modules.sh

module load cuda/11.0.194
module load nccl/2.8.4
#module load intel
module load cudnn/8.1
module load python/3.6.5
module load tensorflow

rootdir=/gs/hs0/tga-shinoda/18B11396/sound-separation/data/Monoral_Isolated_urban_sound/background2
SCRIPT_PATH=/gs/hs0/tga-shinoda/18B11396/sound-separation/models/dcase2020_fuss_baseline/calcdiversity.py
python3 ${SCRIPT_PATH} -nr=${rootdir}