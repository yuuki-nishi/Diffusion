import torch.utils.data
import torchvision.transforms as transforms
import torch.nn as nn
import pathlib
import wave
import numpy as np
import random
import torchaudio
import os
import makeconfig

def getclasslist(pathes):
    ret = [os.path.basename(path)]

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, config,path):
        print("path is")
        print(path)

        self.wav_paths = []
        for c in config.train_data_dir.split(sep=":", maxsplit=-1):
            #tmppath = config.datapath + "/" + c
            tmppath = config.datapath 
            self.wav_paths.extend([str(p) for p in pathlib.Path(tmppath).glob("**/*.wav")])
        #応急的な分割
        tmp = []
        self.samplenum = config.crop_mel_frames*config.hopsize_quant
        self.device = config.device
        self.crop_mel_frames = config.crop_mel_frames
        for p in self.wav_paths:
            #print(p)
            if self.getint(p) >= 2 and not self.filter_shorter(p):#必要以下に短くても消す
                tmp.append(p)
        self.wav_paths = tmp
        self.classlist = [os.path.basename(path)[0:3] for path in self.wav_paths]#ファイルの最初の3字がそれだから
        self.classes = list(set(self.classlist))
        self.class_to_num = {name:i for i, name in enumerate(self.classes)}#string to idx
        self.numed_classes = [self.class_to_num[c] for c in self.classes ]#string to idx
        
        self.classnum = len(self.classes) + 1
        self.classes_one_hot = np.array([np.eye(self.classnum)[c] for c in self.numed_classes])
        self.data_num = len(self.wav_paths)
        #self.config = config
        self.selectclassnum = 1#何個のクラスで学習させるか
        self.config = config
        random.seed(364364)
    def filter_shorter(self,path):
        #cropmelより短いものを消す
        spec = np.asarray(np.load(path + ".spec.npy"))
        #data_len = data.size()[-1]
        return len(spec.T) < self.crop_mel_frames
    def __len__(self):
        return len(self.wav_paths)
    def getint(self,path :str):
        i = os.path.splitext(os.path.basename(path))[0][-1]
        return int(i)
    def pathes_dependonclass(self):
        self.path_class={}#クラスごとのリストを作成
        for c in self.classes:
            self.path_class[c] = []
        
        for path in self.wav_paths:
            c = os.path.basename(path)[0:3]
            self.path_class[c].append(path)
        return

        def __len__(self):
            return self.data_num
    #n sec以内になるように加工
    def transform(self,data,specdata):
        
        # データの読み込み
        #data = wr.readframes(wr.getnframes())
        #data = np.fromstring(data, np.int16) 
        #scale = 1./float(1 << ((8 * bytenum) - 1)) # from librosa
        #data *= scale

        #規定時間内に収まるように切りぬき or 配置
        if data.ndimension() >= 3:
            data = data[0]#次元合わせ
        samplenum=self.samplenum
        framenum = data.size()[-1]
        channelnum = data.size()[-2]
        #print(samplenum)
        #print(len(specdata))
        #print(specdata.shape())
        specdata = np.asarray(specdata)
        data = data[0].numpy()#第一次元が余計にある
        #print("specsize")
        #print(specdata.shape)
        #print(data)
        if len(specdata) >= self.crop_mel_frames:
            #print("ue")
            start = random.randint(0, specdata.shape[0] - self.crop_mel_frames)
            end = start + self.crop_mel_frames
            specdata = specdata[start:end].T
            samples_per_frame = self.config.hopsize_quant
            start *= samples_per_frame
            end *= samples_per_frame
            data = data[start:end]
            #print("len : {}".format(len(data)))
            data = np.pad(data, (0, (end-start) - len(data)), mode='constant')
        else :
            #print("sita")
            raise ValueError("shorter smaple!")
            lspacesize = np.random.randint(0,int(samplenum - framenum))
            rspacesize = samplenum - lspacesize - framenum
            #data shape is [channel, framenum]に注意
            zeros1 = torch.tensor(np.zeros(shape=(channelnum,lspacesize),dtype = float))
            zeros2 = torch.tensor(np.zeros(shape=(channelnum,rspacesize)))
            #print("data test")
            #print(zeros1)
            #print(data)
            #print(zeros2)
            data = torch.cat((zeros1,data,zeros2),dim=1)
            #data = np.zeros(shape=(channelnum,lspacesize),dtype = float).extend(data).extend(np.zeros(shape=(channelnum,rspacesize)),dtype = float)
        #print(data.size())
        #print(soundclass)
        #data = data.float()
        #平均0分散1 にする
        #data = (data-data.mean())/data.std()
        data = data#音量小さくしてみる

        return data,specdata
    def loaddata(self,path):

        data, sample_rate = torchaudio.load(path)
        #data = data.to(self.device)
        #print("raw data size {}".format(data.size()))
        #data shape is [channel, framenum]
        metadata = torchaudio.info(path)
        #wr = wave.open(path, "rb")
        #channelnum = metadata.num_channels
        #print(metadata)
        #samplingrate = metadata.rate
        #framenum = metadata.num_frames
        #bytenum = wav.getsampwidth())#バイト数
        #sec = float(framenum) / samplingrate

        #spectrogram
        spec_data = np.asarray(np.load(path + ".spec.npy"))
        return data,spec_data.T

    def __getitem__(self, idx):
        path = self.wav_paths[idx]
        data,specdata = self.loaddata(path)
        #soundclassはone-hot
        soundclass = self.numed_classes[self.class_to_num[self.classlist[idx]]]
        data,specdata = self.transform(data,specdata)
        specdata = torch.from_numpy(specdata)
        #data=torch.from_numpy(data)
        #specdata=torch.from_numpy(specdata)
        return data,specdata, soundclass

    def sample_classes(self, batch_size):
        ret_data = []
        class_vectors = []
        for i in range(batch_size):
            data_foronebatch = []
            class_foronebatch = []
            for c in self.classes[0:self.selectclassnum] :
                path = random.choice(self.path_class[c])
                class_vector = self.classes_one_hot(self.numed_classes[c])#stringを数値にして、one-hotにする
                rawdata,specdata = self.loaddata(path)
                data,specdata = self.transform(rawdata,specdata)
                data_foronebatch.append(data)
                class_foronebatch.append(class_vector)

            class_vectors.append(class_foronebatch)
            ret_data.append(data_foronebatch)
        return ret_data, class_vectors


