#! /bin/sh
#$ -cwd
#$ -l s_gpu=1
#it was 24:00:00
#$ -l h_rt=24:00:00
#$ -N disc_ev
#$ -t 31-31:1
#$ -o TsubameOutputs/disc_eval/tmp.o
#$ -e TsubameOutputs/disc_eval/tmp.e
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
#./cleartests.sh
#rm -rf /gs/hs0/tga-shinoda/18B11396/sound-separation/models/dcase2020_fuss_baseline/testoutputs/*
noisekinds=("bird" "counstructionSite" "crowd" "foutain" "park" "rain" "schoolyard" "traffic" "ventilation" "wind_tree")
kindint=$(((SGE_TASK_ID-1)%10))
kind=${noisekinds[$kindint]}
pnum=$((((SGE_TASK_ID-1)/10)+1))
echo ${outnum}
paramfile=./inputparams_disc/${kind}_p${pnum}.sh
#pnum=$((SGE_TASK_ID))
#paramfile=./inputparams_disc/p${pnum}.sh
sizeparams=("small" "middle" "large")
#size=${sizeparams[$SGE_TASK_ID-1]}
size=large
#outnum=$((4+(SGE_TASK_ID-1)/10))
#paramfile=./inputparams_disc/bird_p1.sh
echo ${paramfile}

vari_omega=50.0

./run_baseline_dry_model_disc_evaluate.sh ${paramfile} ${size} ${vari_omega}
vari_omega=0.0

./run_baseline_dry_model_disc_evaluate.sh ${paramfile} ${size} ${vari_omega}
