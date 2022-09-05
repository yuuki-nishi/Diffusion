
import tensorflow.compat.v1 as tf
import os
import glob
import copy

from . import inference_graph, InitHook
from . import signal_util

def execute_train(model_dir,
          model_fn,
          params,
          loss_fn,
          input_fn,
          train_input_fn,
          eval_input_fn,
          metrics_fn=None):
  clist = glob.glob(model_dir+"/*ckpt")
  sess = tf.Session()
  saver = tf.compat.v1.train.Saver()
  hparams=params['hparams']
  if len(clist) >0:
    checkpoint_path = tf.train.latest_checkpoint(model_dir)
    tf.train.Saver.restore(sess, checkpoint_path)

  element_from_iterator = train_input_fn()
  # Build the optimizer.
  learning_rate = tf.train.exponential_decay(
      hparams.lr,
      tf.train.get_or_create_global_step(),
      decay_steps=hparams.lr_decay_steps,
      decay_rate=hparams.lr_decay_rate)
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  if params.get('use_tpu', False):
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)
  
  #input = tf.placeholder(tf.float32, name='Input')
  #output = tf.placeholder(tf.float32, name = 'Output')
  for step in range(params["train_steps"]):
      features = sess.run(element_from_iterator)
      sample_mixture=features['receiver_audio']
      sample_source=features['source_images']

      
      #with tf.GradientTape() as tape:
      output = model_fn(sample_mixture,hparams)
      loss = loss_fn(sample_source,output,sample_mixture,hparams)
      print("step : {},loss : {}".format(step,loss))

      if step%20==0:
        train_vars=[]
      else:
        train_vars=[]

      # Build the train_op.
      grads=optimizer.compute_gradients(loss,var_list=train_vars)
      train_op=optimizer.apply_gradients(grads_and_vars=grads)
      sess.run(train_op)

      if step % 2000 == 0:
          # Append the step number to the checkpoint name:
          saver.save(sess, 'model', global_step=step)
