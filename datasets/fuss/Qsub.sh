#! /bin/sh
#$ -cwd
#$ -l s_core=1
#it was 24:00:00
#$ -l h_rt=24:00:00
#$ -N fuss-dataset
. /etc/profile.d/modules.sh

module load cuda/11.0.194
module load nccl/2.8.4
#module load intel
module load cudnn/8.1
module load python/3.6.5
module load tensorflow

#bash run_baseline_model_evaluate.sh
#rm -r data
#rm -r /gs/hs0/tga-shinoda/18B11396/sound-separation/data/*

#export PATH=$PATH:/gs/hs0/tga-shinoda/18B11396/sox-14.4.1
#rm -r ../../data/fuss_augment_2020/Voice
#bash ./run_data_augmentation.sh

python Convert216bit.py