#! /bin/sh
#$ -cwd
#$ -l f_node=1
#it was 24:00:00
#$ -l h_rt=05:00:00
#$ -N f_ef_partgrad
#$ -o TsubameOutputs/sepf_partgrad_multi/tmp_o
#$ -e TsubameOutputs/sepf_partgrad_multi/tmp_e
#$ -t 21-21:1
. /etc/profile.d/modules.sh
module load cuda/11.2.146
module load nccl/2.8.4
#module load intel
module load cudnn/8.1
module load python/3.9.2
module load tensorflow/2.8.0
module list
#python -m pip install attr
#python -m pip install attrs
#pip3 install --user  PySoundFile
#bash run_baseline_dry_model_train.sh
#./cleartests.sh
#rm -rf /gs/hs0/tga-shinoda/18B11396/sound-separation/models/dcase2020_fuss_baseline/testoutputs/*
noisekinds=("bird" "counstructionSite" "crowd" "foutain" "park" "rain" "schoolyard" "traffic" "ventilation" "wind_tree")
kindint=$(((SGE_TASK_ID-1)%10))
kind=${noisekinds[$kindint]}
outnum=$((1+(SGE_TASK_ID-1)/10))

paramfile=./inputparams/${kind}_p${outnum}_${outnum}.sh
#paramfile=./inputparams/bird4.sh


omega=5.0
DiscExecName=${kind}_p${outnum}
echo ${paramfile}
DiscSize=middle
./run_baseline_final_model_evaluate.sh ${paramfile} -part_grad ${DiscExecName} ${omega} ${DiscSize}
