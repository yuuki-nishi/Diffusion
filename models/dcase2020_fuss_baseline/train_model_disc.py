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

import tensorflow as tf

from train import data_io
from train import discmodel
from train import train_with_estimator_disc


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
      '-sz', '--param_size', help='Directory for checkpoints and summaries.',
      required=True)
  parser.add_argument(
      '-cf', '--configfile', help='config file path',
      required=True)
  parser.add_argument(
      '-vari_omega', '--variance_omega', help='config file path', type=float,
      required=True)
  parser.add_argument(
      '-fb', '--final_batch', help='final batch', type=int,
      required=False)
  parser.add_argument(
      '-mixup', '--mixup',action='store_true')
  args = parser.parse_args()
  hparams = discmodel.get_model_hparams()
  hparams.discparam_size=args.param_size
  print("configpath : {}".format(args.configfile))
  print(args.model_dir)
  #hparams.num_sources_for_summaries = [i+1 for i in range(int(args.outnum))]
  hparams.sr = 16000.0
  #hparams.num_sources_for_summaries = [1, 2, 3, 4] #default

  roomsim_params = {
      'num_sources': len(hparams.signal_names),
      'num_receivers': 1,
      'num_samples': int(hparams.sr * 10.0),
  }

  feature_spec = data_io.get_roomsim_spec(**roomsim_params)
  inference_spec = data_io.get_inference_spec_disc()
  
  hparams.classnum=data_io.SoundConfig(sourceroot=args.data_dir,split="train",path=args.configfile).getclassnum_fordisc()
    

  params = {
      'feature_spec': feature_spec,
      'inference_spec': inference_spec,
      'hparams': hparams,
      'io_params': {'parallel_readers': 512,
                    'num_samples': int(hparams.sr * 10.0),
                    'mixup':args.mixup},
      'model_dir': args.model_dir,
      'configpath':args.configfile,
      'source_root': args.data_dir,
      'train_batch_size': args.final_batch*hparams.classnum,
      'eval_batch_size': args.final_batch*hparams.classnum,
      'processed_data_dir': args.p_data_dir,
      #'train_steps': 20000000,
      'train_steps': 200000,
      'train_epoch': 30,
      'train_examples': 10000,
      #'train_examples': 1,
      #'train_examples': 20,
      #'eval_examples': 1000,
      'eval_suffix': 'validation',
      'eval_examples': 200,
      'save_checkpoints_secs': 600,
      'save_summary_steps': 1000,
      'keep_checkpoint_every_n_hours': 4,
      'write_inference_graph': True,
      'randomize_training': True,
      'final_batch': args.final_batch,
      'variance_omega': args.variance_omega,
      #'outnum': int(args.outnum),
  }
  #print(params['input_data_train'])
  discmodel.model_fn(params)


if __name__ == '__main__':
    main()
