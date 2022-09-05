#! /bin/sh
#$ -cwd
#$ -l q_node=1
#it was 24:00:00
#$ -l h_rt=24:00:00
#$ -N disc_tr3
#$ -o TsubameOutputs/disc_multi_train3/tmp_o
#$ -e TsubameOutputs/disc_multi_train3/tmp_e
#$ -t 31-31:1
. /etc/profile.d/modules.sh

module load cuda/11.2.146
module load nccl/2.8.4
#module load intel
module load cudnn/8.1
module load python/3.9.2
module load tensorflow/2.8.0
#python -m pip install attr
#python -m pip install attrs
#bash run_baseline_dry_model_train.sh
sizeparams=("small" "middle" "large")
noisekinds=("bird" "counstructionSite" "crowd" "foutain" "park" "rain" "schoolyard" "traffic" "ventilation" "wind_tree")
kindint=$(((SGE_TASK_ID-1)%10))
kind=${noisekinds[$kindint]}
pnum=$((((SGE_TASK_ID-1)/10)+1))
#pnum=$((SGE_TASK_ID))
echo ${outnum}
#paramfile=./inputparams_disc/${kind}_p${pnum}_zeros.sh
paramfile=./inputparams_disc/${kind}_p${pnum}.sh
#paramfile=./inputparams_disc/p4.sh
size=${sizeparams[$SGE_TASK_ID-1]}
size=large
#outnum=$((4+(SGE_TASK_ID-1)/10))
#paramfile=./inputparams/out${SGE_TASK_ID}.sh
#paramfile=./inputparams_disc/all.sh
echo ${paramfile}
variance_omega=50.0
bash run_baseline_dry_model_disc_train.sh ${paramfile} ${size} ${variance_omega}