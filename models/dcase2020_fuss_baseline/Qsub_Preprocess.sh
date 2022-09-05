#! /bin/sh
#$ -cwd
#$ -l s_core=1
#it was 24:00:00
#$ -l h_rt=24:00:00
#$ -N Preprocess
#$ -o TsubameOutputs/Preprocess/tmp_o
#$ -e TsubameOutputs/Preprocess/tmp_e
#$ -t 1-1:1
. /etc/profile.d/modules.sh

module load cuda/11.2.146
module load nccl/2.8.4
#module load intel
module load cudnn/8.1
module load python/3.9.2
module load tensorflow/2.8.0

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

SCRIPT_PATH=/gs/hs0/tga-shinoda/18B11396/sound-separation/models/dcase2020_fuss_baseline
source ${SCRIPT_PATH}/../../datasets/fuss/setup.sh
source ${SCRIPT_PATH}/setup.sh

OUTPUT_DIR=${MODEL_DIR}/baseline_dry_train_newdisc/variomega_${variance_omega}/${VoiceExec_name}
#rm -rf ${OUTPUT_DIR}


python3 ${SCRIPT_PATH}/datasetpreprocess.py -dd=${DEV_DATA_DIR} \
-md=${OUTPUT_DIR} 
end=$(date +"%T")
echo "Start time: $start, installation end time: $end"
