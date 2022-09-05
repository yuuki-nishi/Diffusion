#! /bin/sh
#$ -cwd
#$ -l q_node=1
#it was 24:00:00
#$ -l h_rt=00:10:00
#$ -N fuss_ev
#$ -o TsubameOutputs/sep_e_nodisc/tmp_o
#$ -e TsubameOutputs/sep_e_nodisc/tmp_e
#$ -t 1-30:1
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
#./cleartests.sh
#rm -rf /gs/hs0/tga-shinoda/18B11396/sound-separation/models/dcase2020_fuss_baseline/testoutputs/*
noisekinds=("bird" "counstructionSite" "crowd" "foutain" "park" "rain" "schoolyard" "traffic" "ventilation" "wind_tree")
kindint=$(((SGE_TASK_ID-1)%10))
kind=${noisekinds[$kindint]}
outnum=$((((SGE_TASK_ID-1)/10)+1))
paramfile=./inputparams/${kind}_p${outnum}_${outnum}.sh
#paramfile=./inputparams/out${SGE_TASK_ID}.sh
#paramfile=./inputparams/bird_p1_1.sh
echo paramfile ${paramfile}
./run_baseline_dry_model_evaluate.sh ${paramfile}