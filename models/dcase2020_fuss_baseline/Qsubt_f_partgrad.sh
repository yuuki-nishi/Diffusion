#! /bin/sh
#$ -cwd
#$ -l s_core=1
#it was 24:00:00
#$ -l h_rt=24:00:00
#$ -N f1_partgrad
#$ -o TsubameOutputs/f_part/tmp_o
#$ -e TsubameOutputs/f_part/tmp_e
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
#bash run_baseline_dry_model_train.sh
#pip3 install --user  attrs
noisekinds=("bird" "counstructionSite" "crowd" "foutain" "park" "rain" "schoolyard" "traffic" "ventilation" "wind_tree")
kindint=$(((SGE_TASK_ID-1)%10))
kind=${noisekinds[$kindint]}
outnum=$((((SGE_TASK_ID-1)/10)+1))
DiscSize=middle
omega=5.0
#10.00が、vars=["sep"]みたいにした方
echo ${outnum}
paramfile=./inputparams/${kind}_p${outnum}_${outnum}.sh
DiscExecName=${kind}_p${outnum}_${DiscSize}
#DiscExecName=bird_p1_small
#paramfile=./inputparams/${kind}_p${outnum}_${outnum}.sh
#paramfile=./inputparams/out${SGE_TASK_ID}.sh
#paramfile=./inputparams/bird_p1_1.sh
echo ${paramfile}
bash run_baseline_dry_model_final_train.sh ${paramfile} -part_grad ${DiscExecName} ${DiscSize} ${omega}