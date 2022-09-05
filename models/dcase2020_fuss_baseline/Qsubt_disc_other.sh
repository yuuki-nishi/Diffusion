#! /bin/sh
#$ -cwd
#$ -l q_node=1
#it was 24:00:00
#$ -l h_rt=24:00:00
#$ -N disc_tr
#$ -o TsubameOutputs/disc_multi_train/tmp_o
#$ -e TsubameOutputs/disc_multi_train/tmp_e
# #$ -t 41-50:1
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
sizeparams=("small" "middle" "large")
#noisekinds=("bird" "counstructionSite" "crowd" "foutain" "park" "rain" "schoolyard" "traffic" "ventilation" "wind_tree")
#kindint=$(((SGE_TASK_ID-1)%10))
#kind=${noisekinds[$kindint]}
#pnum=$((((SGE_TASK_ID-1)/10)+1))
#pnum=$((SGE_TASK_ID))
echo ${outnum}
#paramfile=./inputparams_disc/${kind}_p${pnum}_zeros.sh
paramfile=./inputparams_disc/noises2.sh
#paramfile=./inputparams_disc/p4.sh
size=${sizeparams[$SGE_TASK_ID-1]}
size=small
#outnum=$((4+(SGE_TASK_ID-1)/10))
#paramfile=./inputparams/out${SGE_TASK_ID}.sh
#paramfile=./inputparams_disc/all.sh
echo ${paramfile}
bash run_baseline_dry_model_disc_train.sh ${paramfile} ${size}