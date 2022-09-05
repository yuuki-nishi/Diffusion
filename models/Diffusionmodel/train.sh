#! /bin/sh
#$ -cwd
#$ -l s_gpu=2
#it was 24:00:00
#$ -l h_rt=24:00:00
#$ -o TsubameOutputs/testdiffusetrain/tmp_o
#$ -e TsubameOutputs/testdiffusetrain/tmp_e
#$ -N diffusetrain
source module_load.sh
echo $PATH
which python3
source ./setup.sh
echo "before source" >&2
Train_Data=$TRAIN_SPLIT
Eval_Data=$EVAL_SPLIT
Exec_Name=Test_Epoch30_Channel64_LJ_configed_itspossible
#PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:100
#python -c "help('modules')"
GPU=
python3 trainer.py -tdd=$Train_Data -edd=$Eval_Data -en=$Exec_Name \
    -rt=$EXEC_ROOT -dp=$SPEECH_DATA_PATH