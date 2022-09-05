#!/bin/bash
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

echo "Checking TensorFlow installation..."
if python3 -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"; then
  echo "TensorFlow is working, starting training..."
else
  echo ""
  echo "Before running this script, please install TensorFlow according to the"
  echo "instructions at https://www.tensorflow.org/install/pip."
  exit 0
fi

start=$(date +"%T")
echo "Start time : $start"

SCRIPT_PATH=`dirname $0`

source ${SCRIPT_PATH}/../../datasets/fuss/setup.sh
source ${SCRIPT_PATH}/setup.sh

DATE=`date +%Y-%m-%d_%H-%M-%S`
#OUTPUT_DIR=${MODEL_DIR}/baseline_dry_train/${DATE}
echo $1
source $1
#noiseについて
#5f5rは,5fix値+(0-5)*randomという意味
#VoiceExec_name=${VoiceExec_name}
OUTPUT_DIR=${MODEL_DIR}/baseline_dry_newsep_opsavable/${VoiceExec_name}
#rm ${OUTPUT_DIR}/checkpoint
#OUTPUT_DIR2=${MODEL_DIR}/baseline_dry_train_backup/${VoiceExec_name}
#cp -r ${OUTPUT_DIR} ${OUTPUT_DIR2}
#exit 0
#rm -rf ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}
Preprocessed_data=${SCRIPT_PATH}/../../data/Noise_Voice_Preprocessed
python3 ${SCRIPT_PATH}/train_model.py -dd=${DEV_DATA_DIR} -md=${OUTPUT_DIR} \
 -on=${OutNum} -cf=${ConfigPath} -pdd=${Preprocessed_data}
#-ns オプション は、add as sourceの時につける
#python3 ${SCRIPT_PATH}/train_model.py -dd=${DEV_DATA_DIR}/ssdata -md=${OUTPUT_DIR}

end=$(date +"%T")
echo "Start time: $start, installation end time: $end"
