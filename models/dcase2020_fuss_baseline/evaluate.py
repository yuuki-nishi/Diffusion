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
"""Evaluate separated audio from a DCASE 2020 task 4 separation model."""

import argparse

import evaluate_lib


def main():
  parser = argparse.ArgumentParser(
      description='Evaluate a source separation model.')
  parser.add_argument(
      '-cp', '--checkpoint_path', help='Path for model checkpoint files.',
      required=True)
  parser.add_argument(
      '-mp', '--metagraph_path', help='Path for inference metagraph.',
      required=True)
  parser.add_argument(
      '-dp', '--data_list_path', help='Path for list of files.',
      required=True)
  parser.add_argument(
      '-op', '--output_path', help='Path of resulting csv file.',
      required=True)
  parser.add_argument(
      '-on', '--outnum', help='output num',
      required=True,type=int)
  parser.add_argument(
      '-omega', '--omega', help='omega',
      type=float,default=0.00)
  parser.add_argument(
      '-vn', '--voiceexecname', help='output voiceexecname',
      required=True)
  parser.add_argument(
      '-cf', '--configfile', help='config file path',
      required=True)
  parser.add_argument(
      '-otn', '--outtensorname', help='output tensor name without :0')
  parser.add_argument(
      '-discdir', '--DiscModelDir', help='DiscriminatorModelDir',default= None)
  args = parser.parse_args()
  outnum = args.outnum
  import tensorflow.compat.v1 as tf
  #tf.compat.v1.enable_eager_execution()
  checkpoint_path = tf.train.latest_checkpoint(args.checkpoint_path)

  evaluate_lib.evaluate(checkpoint_path, args.metagraph_path,
                        args.data_list_path, args.output_path, outnum,args.voiceexecname,
                        args.configfile,outtensorname=args.outtensorname,discdir=args.DiscModelDir)


if __name__ == '__main__':
  main()
