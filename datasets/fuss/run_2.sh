#! /bin/sh
#$ -cwd
#$ -l q_node=1
#it was 24:00:00
#$ -l h_rt=24:00:00
#$ -N stereo2monoral
. /etc/profile.d/modules.sh

module load cuda/11.0.194
module load nccl/2.8.4
#module load intel
module load cudnn/8.1
module load python/3.6.5
module load tensorflow
pip install pydub
rm /gs/hs0/tga-shinoda/18B11396/sound-separation/data/Monoral_Isolated_urban_sound/background/*wav
python stereo2monoral.py