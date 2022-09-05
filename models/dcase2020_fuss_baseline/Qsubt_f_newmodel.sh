#! /bin/sh
#$ -cwd
#$ -l q_node=1
#it was 24:00:00
#$ -l h_rt=24:00:00
#$ -N newmodel
#$ -o TsubameOutputs/newfinalmodel_mixup/tmp_o
#$ -e TsubameOutputs/newfinalmodel_mixup/tmp_e
#$ -t 37-57:10
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
#pip3 install --user tensorflow_datasets
noisekinds=("bird" "counstructionSite" "crowd" "foutain" "park" "rain" "schoolyard" "traffic" "ventilation" "wind_tree")
kindint=$(((SGE_TASK_ID-1)%10))
kind=${noisekinds[$kindint]}
outnum=$((((SGE_TASK_ID-1)/10)+1))
DiscSize=middle
omega=$((1*(10**2)))
unit=64
#クラスが4つの時、1e+6が大体境目?
echo ${outnum}

#paramfile=./inputparams/${kind}_p${outnum}_${outnum}.sh
#DiscExecName=${kind}_p${outnum}
paramfile=./inputparams/p${outnum}_${outnum}.sh
DiscExecName=p${outnum}

#DiscExecName=bird_p1_small
#paramfile=./inputparams/${kind}_p${outnum}_${outnum}.sh
#paramfile=./inputparams/out${SGE_TASK_ID}.sh
#paramfile=./inputparams/bird_p1_1.sh
echo ${paramfile}
bash run_newmodel_final_train.sh ${paramfile} -part_grad ${DiscExecName} ${omega} ${unit}
