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

import tensorflow.compat.v1 as tf
import os
import glob
import copy
import time
from . import inference_graph, InitHook
from . import model
from . import discmodel
from . import final_model
from . import signal_util
import keras
import tensorflow

def printvarsfromckpt(path):
  print("print ckpt vars : {}".format(path))
  reader = tf.train.NewCheckpointReader(path)
  var_to_shape_map = reader.get_variable_to_shape_map()
  for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key)) # Remove this is you want to print only variable names
    
def createdictfromckpt(path,str):
  reader = tf.train.NewCheckpointReader(path)
  var_to_shape_map = reader.get_variable_to_shape_map()
  ret = {}
  for key in var_to_shape_map:
    if not "global_steps" in key:
      inkey = str + key.replace(str+"_func/","")
      ret[inkey] = reader.get_tensor(key)

  return ret


def execute(model_fn, input_fn, **params):
  """Execute train or eval and/or inference graph writing.

  Args:
    model_fn: An estimator compatible function taking parameters
              (features, labels, mode, params) that returns a EstimatorSpec.
    input_fn: An estimator compatible function taking 'params' that returns a
              dataset
    **params: Dict of additional params to pass to both model_fn and input_fn.
  """
  print("start exec")
  if params['write_inference_graph']:
    inference_graph.write(model_fn, input_fn, params, params['model_dir'])

  
  hparams = params['hparams']
  # Build the optimizer.
  learning_rate = tf.train.exponential_decay(
      hparams.lr,
      tf.train.get_or_create_global_step(),
      decay_steps=hparams.lr_decay_steps,
      decay_rate=hparams.lr_decay_rate)
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

  #build model
  sep_inputs = keras.Input(shape=(None, 1, 160000))
  with tf.variable_scope("sep"):
    sep_out = model.separate_waveforms(sep_inputs,hparams)
  sepmodel = keras.Model(inputs=sep_inputs, outputs=sep_out, name="sepmodel")
  disc_inputs = keras.Input(shape=(None, 1, 160000))
  fn=lambda t: discmodel.wraped_get_probability(t,hparams)
  with tf.variable_scope("disc"):
    prob_out = tf.map_fn(fn,disc_inputs)
  discmodel = keras.Model(inputs=disc_inputs, outputs=prob_out, name="discmodel")

  @tf.function
  def train_step():
    features = input_fn(params)
    source_waveforms = features['source_images'][:, :, 0]

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      batch_size = signal_util.static_or_dynamic_dim_size(source_waveforms, 0)
      
      num_samples=params["io_params"]["num_samples"]

      mixture_waveforms = features['receiver_audio']
      labels = features['label']
      trainmidnum=500000
      separated_waveforms=sepmodel(mixture_waveforms)
      separated_waveforms = tf.reshape(separated_waveforms,[batch_size,params['outnum'],num_samples])
      probabilities = discmodel(separated_waveforms)
      loss,labelloss,metrics=final_model.calc_loss(separated_waveforms,source_waveforms,probabilities,labels,params)
      sep_vars = sepmodel.trainable_variables
      disc_vars = discmodel.trainable_variables
      
      gradients_of_generator = gen_tape.gradient(loss, sep_vars)
      gradients_of_discriminator = disc_tape.gradient(labelloss, disc_vars)

      
      optimizer.apply_gradients(zip(gradients_of_generator, sep_vars))#この時に自動でglobal_stepが更新される
      optimizer.apply_gradients(zip(gradients_of_discriminator, disc_vars))#この時に自動でglobal_stepが更新される
    return metrics
  checkpoint_dir = params["model_dir"]
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(generator_optimizer=optimizer.copy(),
                                  discriminator_optimizer=optimizer.copy(),
                                  sepmodel=sepmodel,
                                  discmodel=discmodel)
  print("exec2")
  clist = glob.glob(params["model_dir"]+"/*ckpt")
  # specify your saved checkpoint path
  tf.compat.v1.disable_eager_execution()
  if len(clist)>0:
    #modelpath = "/gs/hs0/tga-shinoda/18B11396/sound-separation/models/dcase2020_fuss_baseline/model_data/dcase2020_fuss/baseline_dry_train/Voice4"
    modelpath = params["model_dir"]
    checkpoint_path = tf.train.latest_checkpoint(modelpath)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))
  else:
    #これは初めの一回の時の処理
    trainmidnum=500000
    sep_check_path="{}/model.ckpt-{}".format(params["SepModelDir"],trainmidnum)
    #sep_check_path = tf.train.latest_checkpoint(params["SepModelDir"])
    #sep_check_path = os.path.join(params["SepModelDir"],"model.ckpt-1100690")
    #print_tensors_in_checkpoint_file(sep_check_path, '',True)#そもそもtensorが何もないと出た
    print("disc path pre : {}".format(params["DiscModelDir"]))
    disc_check_path="{}/model.ckpt-{}".format(params["DiscModelDir"],trainmidnum)
    #disc_check_path = tf.train.latest_checkpoint(params["DiscModelDir"])
    print("disc path : {}".format(disc_check_path))
    
    tf.train.init_from_checkpoint(sep_check_path,{"sep_func/":"sep/"})
    tf.train.init_from_checkpoint(disc_check_path,{"disc_func/":"disc/"})
  test_summary_writer = tensorflow.summary.create_file_writer(params["model_dir"]+"/logdir")
  def train():
    global_step = tf.compat.v1.train.get_or_create_global_step()
    while global_step < params["train_steps"]:
      metrics = train_step(global_step)

      if (global_step + 1) % 5000 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)
        with test_summary_writer.as_default():
          for name in metrics.keys():
            tensorflow.summary.scalar(name=name,data=metrics[name],step=tf.compat.v1.train.get_or_create_global_step())

  train()
  print("train end")
