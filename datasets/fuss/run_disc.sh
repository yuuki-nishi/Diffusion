source ./setup.sh
FSD_DIR=${RAW_DATA_DIR}
#rm -rf ${ROOT_DIR}/Noise_Voice_Disc/*
python3 make_disc_examples.py -v ${FSD_DIR} -n ${NOISE_DIR} \
  -o ${ROOT_DIR}/Noise_3Voice_Disc --allow_same 1 --num_train ${NUM_TRAIN} \
  --num_validation ${NUM_VAL} --num_eval ${NUM_EVAL} \
  --random_seed 123