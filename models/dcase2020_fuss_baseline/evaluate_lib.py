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

import os

import numpy as np
from numpy.core import multiarray
import pandas as pd
import tensorflow.compat.v1 as tf
import soundfile
import os
import shutil
import inference
from train import data_io
from train import metrics
from train import permutation_invariant
from train import signal_util

# 各要素の二乗平均の平方根を返す
def _weights_for_nonzero_refs(source_waveforms):
  """Return shape (source,) weights for signals that are nonzero."""
  source_norms = tf.sqrt(tf.reduce_mean(tf.square(source_waveforms), axis=-1))
  return tf.greater(source_norms, 1e-8)

# 一番小さい音から計算している
def _weights_for_active_seps(power_sources, power_separated):
  """Return (source,) weights for active separated signals."""
  min_power = tf.reduce_min(power_sources, axis=-1, keepdims=True)
  return tf.greater(power_separated, 0.5 * min_power)
  #return tf.greater(power_separated, 0.5 * min_power)

def printvars(path):
  print("vars from {}".format(path))
  #variables = tf.train.list_variables(path)
  ckpt_reader = tf.train.NewCheckpointReader(path)
  for v in  ckpt_reader.get_variable_to_shape_map():
    value = ckpt_reader.get_tensor(v)
    #print(" {} : {}".format(v,value))
    print(" {}".format(v))
  print("end")
  return
def printvars_meta(path):
  print("vars from meta {}".format(path))
  tf.train.import_meta_graph(path)
  for n in tf.get_default_graph().as_graph_def().node:
    print(n)
  print("end")
  return
def compute_metrics(source_waveforms, separated_waveforms, mixture_waveform,label,config,permed_prob):
  """Permutation-invariant SI-SNR, powers, and under/equal/over-separation."""

  # Align separated sources to reference sources.
  #print(source_waveforms.shape)
  #print(separated_waveforms.shape)
  perm_inv_loss = permutation_invariant.wrap(
      lambda tar, est: -metrics.signal_to_noise_ratio_gain_invariant(est, tar))
  _, separated_waveforms = perm_inv_loss(source_waveforms[tf.newaxis],
                                        separated_waveforms[tf.newaxis])
  """else:
    #print("oooookkkkkkk")#ok
    perm_inv_loss = permutation_invariant.wrap_label_useonlystandard(
        lambda tar, est: -metrics.signal_to_noise_ratio_gain_invariant(est, tar),
        lambda lprob,llabel: np.sum(np.multiply(lprob,llabel)))
    _, separated_waveforms,_ = perm_inv_loss(source_waveforms[tf.newaxis],
                                                              separated_waveforms[tf.newaxis],
                                                              [label],#add batch axis
                                                              [prob])"""
  sourcenum=signal_util.static_or_dynamic_dim_size(source_waveforms, 0)
  samplenum=signal_util.static_or_dynamic_dim_size(source_waveforms, -1)
  #print(sourcenum)
  #print(samplenum)
  #separated_waveforms=tf.reshape(separated_waveforms,[sourcenum,1,samplenum])
    

  separated_waveforms = separated_waveforms[0]  # Remove batch axis.
  """if discmodel is not None:
    reshaped=tf.reshape(separated_waveforms,[sourcenum,1,samplenum])
    permed_prob=discmodel.getprobability(reshaped)"""
    
  # Compute separated and source powers.
  power_separated = tf.reduce_mean(separated_waveforms ** 2, axis=-1)
  power_sources = tf.reduce_mean(source_waveforms ** 2, axis=-1)

  # Compute weights for active (separated, source) pairs where source is nonzero
  # and separated power is above threshold of quietest source power - 20 dB.
  weights_active_refs = _weights_for_nonzero_refs(source_waveforms)
  weights_active_seps = _weights_for_active_seps(#出力された音が無音かどうかを判定している(minの閾値には正解のものを使っている)
      tf.boolean_mask(power_sources, weights_active_refs), power_separated)
  
  weights_active_pairs = tf.logical_and(weights_active_refs,
                                        weights_active_seps)

  # Compute SI-SNR.
  sisnr_separated = metrics.signal_to_noise_ratio_gain_invariant(
      separated_waveforms, source_waveforms)
  num_active_refs = tf.reduce_sum(tf.cast(weights_active_refs, tf.int32))
  num_active_seps = tf.reduce_sum(tf.cast(weights_active_seps, tf.int32))
  num_active_pairs = tf.reduce_sum(tf.cast(weights_active_pairs, tf.int32))
  sisnr_mixture = metrics.signal_to_noise_ratio_gain_invariant(
      tf.tile(mixture_waveform[tf.newaxis], (source_waveforms.shape[0], 1)),
      source_waveforms)

  # Compute under/equal/over separation.
  under_separation = tf.cast(tf.less(num_active_seps, num_active_refs),
                             tf.float32)
  equal_separation = tf.cast(tf.equal(num_active_seps, num_active_refs),
                             tf.float32)
  over_separation = tf.cast(tf.greater(num_active_seps, num_active_refs),
                            tf.float32)


  retlabeldict = {}
  for k in label:
    retlabeldict[config.getname(k)] = []
  for i in range(len(label)):
    truelabel=config.getname(label[i])#これはstring
    retlabeldict[truelabel].append(sisnr_separated[i])#ラベルごとのスコア
  
  #print(permed_prob.shape)
  #print(label.shape)
  retaccdict = {}
  prob2=tf.reshape(permed_prob,label.shape)
  for k in label:
    retaccdict[config.getname(k)] = []
  atetadict={}
  forcoeff={}
  for i in range(sourcenum):
    truelabel=config.getname(label[i])#これはstring
    labelindex=np.argmax(label[i])
    probargmaxindex=np.argmax(prob2[i])
    if probargmaxindex == labelindex:
      atetadict[truelabel]= 1
    else:
      atetadict[truelabel] = 0
    retaccdict[truelabel].append(prob2[i][labelindex].numpy())#ラベルごとの確率
    #forcoeff[truelabel].append([prob2[i][labelindex].numpy(),retlabeldict[truelabel]])
  ret={'sisnr_separated': sisnr_separated,
        'sisnr_mixture': sisnr_mixture,
        'sisnr_improvement': sisnr_separated - sisnr_mixture,
        'power_separated': power_separated,
        'power_sources': power_sources,
        'under_separation': under_separation,
        'equal_separation': equal_separation,
        'over_separation': over_separation,
        'weights_active_refs': weights_active_refs,
        'weights_active_seps': weights_active_seps,
        'weights_active_pairs': weights_active_pairs,
        'num_active_refs': num_active_refs,
        'num_active_seps': num_active_seps,
        'num_active_pairs': num_active_pairs,
        'accuracy':retaccdict,
        'atetadict':atetadict,
        'scoreperlabel':retlabeldict}
  return ret


def _report_score_stats(metric_per_source_count, label='', counts=None):
  """Report mean and std dev for specified counts."""
  values_all = []
  if counts is None:
    counts = metric_per_source_count.keys()
  for count in counts:
    values = metric_per_source_count[count]
    values_all.extend(list(values))
  return '%s for count(s) %s = %.2f +/- %.2f dB' % (
      label, counts, np.mean(values_all), np.std(values_all))


def write_waveforms(mixed,waveforms,index,outpath,source):
  if index >= 16:
    return
  waveforms = waveforms/max(waveforms.min(), waveforms.max(), key=abs)
  mixed = mixed/max(mixed.min(), mixed.max(), key=abs)
  source = source/max(source.min(), source.max(), key=abs)
  path = "/gs/hs0/tga-shinoda/18B11396/sound-separation/models/dcase2020_fuss_baseline/testoutputs/"+outpath
  if not os.path.exists(path):
      # ディレクトリが存在しない場合、ディレクトリを作成する
      os.makedirs(path)
  num = tf.shape(waveforms)[0]
  sourcenum = tf.shape(source)[0]
  path2=path+"/{}".format(index)
  if os.path.exists("{}/{}".format(path,index)):
    shutil.rmtree("{}/{}".format(path,index))
  for i in range(num):
    outputpath="{}/{}/{}.wav".format(path,index,i)
    print("write {}".format(outputpath))
    voice = waveforms[i]
 
    SAMPLE_DIR = "{}/{}".format(path,index)
    
    if not os.path.exists(SAMPLE_DIR):
        # ディレクトリが存在しない場合、ディレクトリを作成する
        os.makedirs(SAMPLE_DIR)
    soundfile.write(outputpath, voice, 16000, subtype='PCM_16')
    
  for i in range(sourcenum):
    outputpath2="{}/{}/s{}.wav".format(path,index,i)
    soundfile.write(outputpath2, source[i], 16000, subtype='PCM_16')
  #print(mixed.shape)
  outputpath="{}/{}/mix.wav".format(path,index)
  soundfile.write(outputpath, mixed, 16000, subtype='PCM_16')
  return

def evaluate(checkpoint_path, metagraph_path, data_source_dir, output_path,
  outnum,exec_name,configpath,outtensorname=None,discdir=None):
  """Evaluate a model on FUSS data."""
  model = inference.FinalSeparationModel(checkpoint_path, metagraph_path,outtensorname)
  print("checkpath : {}".format(checkpoint_path))
  print("metapath : {}".format(metagraph_path))
  print("datapath : {}".format(data_source_dir))
  Config = data_io.SoundConfig(data_source_dir,"eval",configpath)
  with model.graph.as_default():
    dataset = data_io.wavs_to_dataset(configpath,
                                      "eval",
                                      data_source_dir,
                                      batch_size=1,
                                      #datanum=10000,
                                      datanum=1000,
                                      num_samples=160000,
                                      repeat=False,
                                      max_sources_override = outnum)# attention
    # Strip batch and mic dimensions.
    dataset['receiver_audio'] = dataset['receiver_audio'][0, 0]
    dataset['source_images'] = dataset['source_images'][0, :, 0]
  isdisc=not(discdir is None)
  #isdisc=False
  if isdisc:
    disc_check_path = tf.train.latest_checkpoint(discdir)
    print("disc path : {}".format(disc_check_path))
    disc_meta_path = discdir+"/inference.meta"
    #discmodel = inference.DiscriminateModel(disc_check_path, disc_meta_path)
  print(dataset)
  print("tmpshape")
  print(dataset['source_images'].shape)
  print(len(list(dataset)))

  # Separate with a trained model.
  i = 1
  max_count = outnum
  dict_per_source_count = lambda: {c: [] for c in range(1, max_count + 1)}
  sisnr_per_source_count = dict_per_source_count()
  sisnri_per_source_count = dict_per_source_count()
  under_seps = []
  equal_seps = []
  over_seps = []
  outprobs=[]
  accdict={}
  kinddict={}
  atetadict={}
  df = None
  while True:
    print(i)
    try:
      waveforms = model.sess.run(dataset)
    except tf.errors.OutOfRangeError:
      print("i = {},out of range".format(i))
      break
    separated_waveforms= model.getwavs(waveforms['receiver_audio'],waveforms['label'])
    source_waveforms = waveforms['source_images']
    print(source_waveforms.shape)
    separated_waveforms=tf.reshape(separated_waveforms,tf.shape(source_waveforms))
    frommodel_probs=model.getprobs(separated_waveforms)
    label=waveforms['label'][0]


    forshuffle_matrix = tf.eye(outnum)
    forshuffle_matrix=tf.random.shuffle(forshuffle_matrix)#並べ替え用の行列
    int_forshuffle_matrix=tf.cast(forshuffle_matrix,dtype=label.dtype)#並べ替え用の行列
    shuffle_fn=lambda t: tf.matmul(forshuffle_matrix,t)
    int_shuffle_fn=lambda t: tf.matmul(int_forshuffle_matrix,t)
    source_waveforms = shuffle_fn(source_waveforms)
    label = int_shuffle_fn(label)


    labelindex = np.argmax(label)
    #write_waveforms(waveforms['receiver_audio'],separated_waveforms,i,exec_name,source_waveforms)
    tmp_source_waveforms=tf.reshape(separated_waveforms,[tf.size(separated_waveforms)/160000,1,160000])
    probs_from_source=model.getprobs(tmp_source_waveforms)
    probs_from_source=tf.squeeze(probs_from_source)
    print("probs_from_model")
    sourcenum=frommodel_probs.shape[-1]
    probs_from_model=tf.reshape(frommodel_probs,[sourcenum,sourcenum])
    for i in range(sourcenum):
      print("prob {} : {}".format(i,probs_from_model[i]))
    print("probs_from_source")
    fromsource_probs=model.getprobs(source_waveforms)
    probs_from_source=tf.reshape(fromsource_probs,[sourcenum,sourcenum])
    for i in range(sourcenum):
      print("prob {} : {}".format(i,probs_from_source[i]))

    #print("shape : {},{}".format(separated_waveforms.shape,source_waveforms.shape)) #okだった
    if np.allclose(source_waveforms, 0):
      print('WARNING: all-zeros source_waveforms tensor encountered.'
            'Skiping this example...')
      continue
    if isdisc:
      #print("shape label : {}".format(label))
      #print("shape prob : {}".format(prob))
      metrics_dict = compute_metrics(source_waveforms, separated_waveforms,
                                   waveforms['receiver_audio'],label,Config,probs_from_source)
    else:
      metrics_dict = compute_metrics(source_waveforms, separated_waveforms,
                                   waveforms['receiver_audio'],label,Config)

    metrics_dict = {k: v.numpy() if hasattr(v,"numpy") else v for k, v in metrics_dict.items()}#numpy化できるものだけ
    sisnr_sep = metrics_dict['sisnr_separated']
    sisnr_mix = metrics_dict['sisnr_mixture']
    sisnr_imp = metrics_dict['sisnr_improvement']
    weights_active_pairs = metrics_dict['weights_active_pairs']

    if isdisc:
      #accuracys.append(metrics_dict["accuracy"])
      acctmpdict = metrics_dict["accuracy"]
      tmpatetadict = metrics_dict["atetadict"]

      outprobs.append( np.mean(np.max(frommodel_probs,axis=-1)))
      for k in acctmpdict.keys():
        #kは種類名
        if k in accdict:
          accdict[k].extend(acctmpdict[k])
          atetadict[k]+=tmpatetadict[k]
        else:
          accdict[k]=acctmpdict[k]
          atetadict[k]=tmpatetadict[k]

    kindtmpdict = metrics_dict["scoreperlabel"]
    for k in kindtmpdict.keys():
      if k in kinddict:
        kinddict[k].extend(kindtmpdict[k])
      else:
        kinddict[k]=kindtmpdict[k]
    """if i>=5:
      return"""
    # Create and initialize the dataframe if it doesn't exist.
    if df is None:
      # Need to create the dataframe.
      columns = []
      """for metric_name, metric_value in metrics_dict.items():
        if metric_name=="scoreperlabel" or metric_name=="accuracy":
          continue
        if metric_value.shape:
          # Per-source metric.
          for i_src in range(1, max_count + 1):
            columns.append(metric_name + '_source%d' % i_src)
        else:
          # Scalar metric.
          columns.append(metric_name)"""
      columns.sort()
      df = pd.DataFrame(columns=columns)
      if output_path.endswith('.csv'):
        csv_path = output_path
      else:
        csv_path = os.path.join(output_path, 'scores.csv')

    # Update dataframe with new metrics.
    row_dict = {}
    new_row = pd.Series(row_dict,dtype=float)
    df = pd.concat([df,new_row], ignore_index=True)

    # Store metrics per source count and report results so far.
    under_seps.append(metrics_dict['under_separation'])
    equal_seps.append(metrics_dict['equal_separation'])
    over_seps.append(metrics_dict['over_separation'])
    sisnr_per_source_count[metrics_dict['num_active_refs']].extend(
        sisnr_sep[weights_active_pairs].tolist())
    sisnri_per_source_count[metrics_dict['num_active_refs']].extend(
        sisnr_imp[weights_active_pairs].tolist())
    print('Example %d: SI-SNR sep = %.1f dB, SI-SNR mix = %.1f dB, '
          'SI-SNR imp = %.1f dB, ref count = %d, sep count = %d' % (
              i, np.mean(sisnr_sep), np.mean(sisnr_mix),
              np.mean(sisnr_sep - sisnr_mix), metrics_dict['num_active_refs'],
              metrics_dict['num_active_seps']))
    i += 1

  # Report final mean statistics.
  
  lines = ["Final statistics"]
  lines.append(["tested num : {}".format(i)])
  for i in range(1,outnum+1):
    lines.append(_report_score_stats(sisnr_per_source_count, 'SI-SNR',
                          counts=[i]))
  lines.append(_report_score_stats(sisnri_per_source_count, 'SI-SNRi',
                          counts=[i for i in range(2,outnum+1)]))
  lines.append('Under separation: %.2f' % np.mean(under_seps))
  lines.append('Equal separation: %.2f' % np.mean(equal_seps))
  lines.append('Over separation: %.2f' % np.mean(over_seps))
  for k in kinddict.keys():
    lines.append("{} : {} +/- {}".format(k,np.mean(kinddict[k]),np.std(kinddict[k])))
  print("accuracies")
  if isdisc:
    lines.append("frommodel_probs: %.2f" % np.mean(outprobs))
    for k in accdict.keys():
      #print(accdict)
      #print("key : {}".format(k))
      lines.append("mean acc {} : {} +/- {}".format(k,np.mean(accdict[k]),np.std(accdict[k])))
      coef=np.corrcoef([accdict[k],kinddict[k]])
      lines.append("coef {} : {}".format(k,coef))
    for k in accdict.keys():
      lines.append("accuracy {} : {}".format(k,atetadict[k]/10000))
  
  for line in lines:
    print(line)
  with open(csv_path.replace('.csv', '_summary.txt'), 'w+') as f:
    f.writelines([str(line) + '\n' for line in lines])

  # Write final csv.
  print('\nWriting csv to %s.' % csv_path)
  df.to_csv(csv_path)
