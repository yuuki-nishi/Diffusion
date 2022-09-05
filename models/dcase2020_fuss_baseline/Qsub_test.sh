#! /bin/sh
#$ -cwd
#$ -l q_node=1
#it was 24:00:00
#$ -l h_rt=1:00:00
#$ -N test
#$ -o TsubameOutputs/test/tmp_e
#$ -e TsubameOutputs/test/tmp_o
. /etc/profile.d/modules.sh

module load cuda/11.0.194
module load nccl/2.8.4
#module load intel
module load cudnn/8.1
module load python/3.6.5
module load tensorflow

#rm -r /gs/hs0/tga-shinoda/18B11396/sound-separation/models/dcase2020_fuss_baseline/testoutputs
cp -r /gs/hs0/tga-shinoda/18B11396/sound-separation/models/dcase2020_fuss_baseline/model_data/dcase2020_fuss/baseline_final_train_disctrain2 /gs/hs0/tga-shinoda/18B11396/sound-separation/models/dcase2020_fuss_baseline/model_data/dcase2020_fuss/baseline_final_train_backup
#cp -r /gs/hs0/tga-shinoda/18B11396/sound-separation/models/dcase2020_fuss_baseline/model_data/dcase2020_fuss/baseline_final_train_3stepdiscsepdisep /gs/hs0/tga-shinoda/18B11396/sound-separation/models/dcase2020_fuss_baseline/model_data/dcase2020_fuss/baseline_final_train_recstepbackup