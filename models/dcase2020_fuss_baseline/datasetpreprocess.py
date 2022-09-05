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
import copy
import random
import tensorflow as tf
import typing as tp
from train import data_io
from train import final_model
from train import train_with_estimator_final
from train import sounddata
import numpy as np
import soundfile as sf
def execute(params,split):
    
    DataNumperClass = 10000
    noiseclass = ["bird","counstructionSite","crowd","foutain","park","rain","schoolyard","traffic","ventilation","wind_tree"]
    needpersion=sounddata.needpersion
    Processed_data_dir = params["source_root"] + "/../Noise_Voice_Preprocessed"
    classes=[]
    print(sounddata.noiseclass)
    classes.extend(noiseclass)
    classes.extend(needpersion)
    print(classes)
    dataset = data_io.SoundConfig(params["source_root"],split,classes=classes)
    tf.random.set_seed(114514)
    random.seed(114514)
    print(dataset)
    _format = "WAV"
    for c in classes:
        write_dir = Processed_data_dir + "/{}/{}".format(c,split)
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)
        pathes = dataset.SoundPathDict[c]
        for i in range(DataNumperClass):
            data=[]
            while True:#音量が0にならないまでやる
                nowpath=random.choice(pathes)
                decoded_wav=data_io.decode_wav(nowpath)
                decoded_wav=data_io.scaling(decoded_wav,data_io.setdb(c))
                var = tf.math.reduce_variance(decoded_wav).numpy()
                if var < 0.01:
                    print("continue by var")
                    print(nowpath)
                    print(decoded_wav.shape)
                    continue
                else:
                    data=decoded_wav
                    break
            out_path = write_dir+"/{}.wav".format(i)
            wave = np.reshape(data,[160000,1])
            sf.write(out_path, wave, 16000, format=_format)
             
    return


def main():
    parser = argparse.ArgumentParser(
        description='Train the DCASE2020 FUSS baseline source separation model.')
    parser.add_argument(
        '-dd', '--data_dir', help='Data directory.',
        required=True)
    parser.add_argument(
        '-md', '--model_dir', help='Directory for checkpoints and summaries.',
        required=True)
    args = parser.parse_args()
    
    
    hparams = final_model.get_model_hparams_withOutNum(5)
    print(hparams)
    hparams.sr = 16000.0
    
    params = {
        'hparams': hparams,
        'io_params': {'parallel_readers': 512,
                        'num_samples': int(hparams.sr * 10.0)},
        'source_root': args.data_dir,
        'model_dir': args.model_dir,
    }
    splits = ["train","eval"]
    for split in splits:
        execute(params,split)
    return


if __name__ == '__main__':
    main()
