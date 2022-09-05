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
r"""Train the DCASE2020 FUSS baseline source separation model."""

import argparse
import os

import tensorflow.compat.v1 as tf

from train import data_io,model
#from train_v1 import data_io,model,train_with_estimator,train_with_loop


def main():
  parser = argparse.ArgumentParser(
      description='Train the DCASE2020 FUSS baseline source separation model.')
  parser.add_argument(
      '-dd', '--data_dir', help='Data directory.',
      required=True)
  parser.add_argument(
      '-pdd', '--p_data_dir', help='Processed Data directory.',
      required=True)
  parser.add_argument(
      '-md', '--model_dir', help='Directory for checkpoints and summaries.',
      required=True)
  parser.add_argument(
      '-cf', '--configfile', help='config file path',
      required=True)
  parser.add_argument(
      '-on', '--outnum', help='Output num',
      required=True,type=int)
  args = parser.parse_args()
  print("eager : {}".format(tf.executing_eagerly()))
  print(args.outnum)
  outnum = args.outnum
  print(outnum)
  hparams = model.get_model_hparams_withOutNum(outnum)
  print(hparams.signal_names)
  hparams.num_sources_for_summaries = [i+1 for i in range(outnum)]
  hparams.sr = 16000.0

  roomsim_params = {
      'num_sources': len(hparams.signal_names),
      'num_receivers': 1,
      'num_samples': int(hparams.sr * 10.0),
  }
  tf.logging.info('Params: %s', roomsim_params.values())
  classnum=data_io.SoundConfig(sourceroot=args.data_dir,split="train",path=args.configfile).MaxNum
  feature_spec = data_io.get_roomsim_spec(**roomsim_params)
  inference_spec = data_io.get_inference_spec(classnum=classnum)
  

  tf.debugging.set_log_device_placement(True)
  params = {
      'feature_spec': feature_spec,
      'inference_spec': inference_spec,
      'hparams': hparams,
      'io_params': {'parallel_readers': 512,
                    'num_samples': int(hparams.sr * 10.0)},
      'source_root': args.data_dir,
      'model_dir': args.model_dir,
      'configpath':args.configfile,
      'train_batch_size': 4,
      'eval_batch_size': 4,
      #'train_steps': 20000000,
      'train_steps': 100000,
      'processed_data_dir': args.p_data_dir,
      #'train_steps': 100,
      'eval_suffix': 'validation',
      'eval_examples': 100,
      'train_examples': 10000,
      'save_checkpoints_secs': 600,
      'save_summary_steps': 1000,
      'keep_checkpoint_every_n_hours': 1,
      'randomize_training': True,
      'save_checkpoints_steps':50000,
      #'part_grad':args.part_grad,#部分的な勾配の更新をするか
      'outnum': int(args.outnum),
  }
  tf.logging.info(params)
  model.model_fn(params)


if __name__ == '__main__':
    main()
