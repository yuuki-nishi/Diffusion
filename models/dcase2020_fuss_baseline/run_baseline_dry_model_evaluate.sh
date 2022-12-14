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
  echo "TensorFlow is working, starting evaluation..."
else
  echo ""
  echo "Before running this script, please install TensorFlow according to the"
  echo "instructions at https://www.tensorflow.org/install/pip."
  exit 0
fi

start=$(date +"%T")
echo "Start time : $start"

SCRIPT_PATH=`dirname $0`

echo $1
source $1
source ${SCRIPT_PATH}/../../datasets/fuss/setup.sh
source ${SCRIPT_PATH}/setup.sh

bash ${SCRIPT_PATH}/install_dependencies.sh

bash ${SCRIPT_PATH}/get_pretrained_baseline_dry_model.sh

#VoiceExec_name=${VoiceExec_name}_3Voice
#Needns=-ns
#noiseについて
#5f5rは,5fix値+(0-5)*randomという意味
#VoiceExec_name=${VoiceExec_name}_0f_5r
#BASELINE_DRY_MODEL_DIR=${BASELINE_DRY_MODEL_DIR}

DATE=`date +%Y-%m-%d_%H-%M-%S`
#OUTPUT_DIR=${MODEL_DIR}/baseline_dry_evaluate/${DATE}
OUTPUT_DIR=${MODEL_DIR}/baseline_dry_evaluate/${VoiceExec_name}
mkdir -p ${OUTPUT_DIR}
#python3 ${SCRIPT_PATH}/evaluate.py -cp=${BASELINE_DRY_MODEL_DIR}/baseline_dry_model -mp=${BASELINE_DRY_MODEL_DIR}/baseline_dry_inference.meta -dp=${DEV_DATA_DIR}/ssdata/eval_example_list.txt -op=${OUTPUT_DIR}

#python3 ${SCRIPT_PATH}/evaluate.py -cp=${BASELINE_DRY_MODEL_DIR}/model.ckpt-1000000 -mp=${BASELINE_DRY_MODEL_DIR}/inference.meta -dp=${DEV_DATA_DIR}/Voice/eval_example_list.txt -op=${OUTPUT_DIR} #こんな感じで訓練したモデルをevaluateする
python3 ${SCRIPT_PATH}/evaluate.py -cp=${BASELINE_DRY_MODEL_DIR} -mp=${BASELINE_DRY_MODEL_DIR}/inference.meta \
-dp=${DEV_DATA_DIR} -op=${OUTPUT_DIR} \
-on=${OutNum} -vn=${VoiceExec_name} -cf=${ConfigPath} \
-otn=sep_func/seped_waveforms
end=$(date +"%T")
echo "Start time: $start, installation end time: $end"
