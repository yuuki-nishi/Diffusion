#! /bin/sh
#$ -cwd
#$ -l q_node=1
#it was 24:00:00
#$ -l h_rt=01:00:00
#$ -N fuss_ef
#$ -o TsubameOutputs/sepf_allgrad/tmp_o
#$ -e TsubameOutputs/sepf_allgrad/tmp_e
# #$ -t 1-20:1
. /etc/profile.d/modules.sh

module load cuda/11.0.194
module load nccl/2.8.4
#module load intel
module load cudnn/8.1
module load python/3.6.5
module load tensorflow
#python -m pip install attr
#python -m pip install attrs
#bash run_baseline_dry_model_train.sh
./cleartests.sh
#rm -rf /gs/hs0/tga-shinoda/18B11396/sound-separation/models/dcase2020_fuss_baseline/testoutputs/*
noisekinds=("bird" "counstructionSite" "crowd" "foutain" "park" "rain" "schoolyard" "traffic" "ventilation" "wind_tree")
kind=${noisekinds[$SGE_TASK_ID%10]}
outnum=$((4+(SGE_TASK_ID-1)/10))
#paramfile=./inputparams/${kind}${outnum}.sh
#paramfile=./inputparams/out${SGE_TASK_ID}.sh
paramfile=./inputparams/bird4.sh
echo ${paramfile}
./run_baseline_final_model_evaluate.sh ${paramfile}