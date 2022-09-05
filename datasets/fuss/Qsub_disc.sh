#! /bin/sh
#$ -cwd
#$ -l s_core=1
#it was 24:00:00
#$ -l h_rt=2:00:00
#$ -N disc_makedatasets
. /etc/profile.d/modules.sh

module load cuda/11.0.194
module load nccl/2.8.4
#module load intel
module load cudnn/8.1
module load python/3.6.5
#module load tensorflow
#pip install 
source ./setup.sh
rm -r ${ROOT_DIR}/Noise_Voice_Disc
echo ${NUM_TRAIN}
FSD_DIR=${RAW_DATA_DIR}
python3 make_disc_examples.py -v ${FSD_DIR} -n ${NOISE_DIR} \
  -o ${ROOT_DIR}/Noise_3Voice_Disc --allow_same 1 --num_train ${NUM_TRAIN} \
  --num_validation ${NUM_VAL} --num_eval ${NUM_EVAL} \
  --random_seed 123