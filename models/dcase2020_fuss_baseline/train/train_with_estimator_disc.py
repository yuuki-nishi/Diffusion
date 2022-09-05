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
"""Train helper for source separation using tf.estimator."""

import tensorflow as tf
import os
import glob

from . import inference_graph, InitHook


def execute(model_fn, input_fn, **params):
  """Execute train or eval and/or inference graph writing.

  Args:
    model_fn: An estimator compatible function taking parameters
              (features, labels, mode, params) that returns a EstimatorSpec.
    input_fn: An estimator compatible function taking 'params' that returns a
              dataset
    **params: Dict of additional params to pass to both model_fn and input_fn.
  """
  if params['write_inference_graph']:
    print("inference_graph_mode")
    #inference_graph.write(model_fn, input_fn, params, params['model_dir'])

  def estimator_model_fn(features, labels, mode, params):
    spec = model_fn(features, labels, mode, params)
    return spec
  def train_input_fn():
    train_params = params.copy()
    train_params['batch_size'] = params['train_batch_size']
    if params['randomize_training']:
      train_params['randomize_order'] = True
    train_params["split"]="train"
    train_params["example_num"]=params["train_examples"]
    return input_fn(train_params)

  def eval_input_fn():
    eval_params = params.copy()
    eval_params['batch_size'] = params['eval_batch_size']
    eval_params["split"]="eval"
    eval_params["example_num"]=params["eval_examples"]
    return input_fn(eval_params)
  
  train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                      max_steps=params['train_steps'])

  eval_steps = int(round(params['eval_examples'] / params['eval_batch_size']))

  eval_spec = tf.estimator.EvalSpec(
      name=params['eval_suffix'], input_fn=eval_input_fn, steps=eval_steps,
      throttle_secs=params.get('eval_throttle_secs', 600))

  run_config = tf.estimator.RunConfig(
      model_dir=params['model_dir'],
      save_summary_steps=params['save_summary_steps'],
      save_checkpoints_secs=params['save_checkpoints_secs'],
      keep_checkpoint_every_n_hours=params['keep_checkpoint_every_n_hours'])
  clist = glob.glob(params["model_dir"]+"/*ckpt")
  # specify your saved checkpoint path
  """if len(clist)>0:
    #modelpath = "/gs/hs0/tga-shinoda/18B11396/sound-separation/models/dcase2020_fuss_baseline/model_data/dcase2020_fuss/baseline_dry_train/Voice4"
    modelpath = params["model_dir"]
    checkpoint_path = tf.train.latest_checkpoint(modelpath)

    #checkpoint_path = "./model_data/dcase2020_fuss/baseline_dry_train/2021-07-17_17-04-23/"
    # ref : https://stackoverflow.com/questions/49846207/tensorflow-estimator-warm-start-from-and-model-dir
    #print(checkpoint_path)
    ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from = checkpoint_path)
    inithook = InitHook.InitHook(checkpoint_dir = modelpath)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                      max_steps=params['train_steps'],hooks = [inithook])
    estimator = tf.estimator.Estimator(
        model_fn=estimator_model_fn,
        params=params,
        config=run_config,
        model_dir = params['model_dir'],
        warm_start_from=ws#I added
        )
  else:
    estimator = tf.estimator.Estimator(
        model_fn=estimator_model_fn,
        params=params,
        model_dir = params['model_dir'],
        config=run_config
        )"""
  print("start train")
  #tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  model_fn(features, labels, params)
  print("train end")
