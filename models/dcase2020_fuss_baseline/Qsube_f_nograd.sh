#! /bin/sh
#$ -cwd
#$ -l q_node=1
#it was 2:00:00
#$ -l h_rt=02:00:00
#$ -N fuss_ef_nograd
#$ -o TsubameOutputs/sepf_nograd_multi/tmp.o
#$ -e TsubameOutputs/sepf_nograd_multi/tmp.e
#$ -t 31-40:1
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
DiscSize=small
echo ${outnum}
paramfile=./inputparams/${kind}_p${outnum}_${outnum}.sh

omega=5.0
DiscExecName=${kind}_p${outnum}_${DiscSize}
echo ${paramfile}
./run_baseline_final_model_evaluate.sh ${paramfile} -nograd ${DiscExecName} ${omega}

