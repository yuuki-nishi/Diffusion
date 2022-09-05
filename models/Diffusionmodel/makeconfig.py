import torch
import numpy as np
import torch.nn as nn
ConvTranspose2d = nn.ConvTranspose2d
class Myconfig():
    def __init__(self,args):
        self.checkpointpath = args.root + "/model_data/" + args.exec_name
        self.datapath = args.datapath
        self.train_data_dir = args.train_data_dir
        self.eval_data_dir = args.eval_data_dir
        self.samplingrate = 22050
        self.wavdirpath =  args.root + "/model_data/" +"{}/WavSample".format(args.exec_name)
        self.lossgraphpath = args.root + "/model_data/" +"{}/plotgraph.png".format(args.exec_name)
        self.stdplotpath = args.root + "/model_data/" +"{}/stdplot.png".format(args.exec_name)
        self.loggerpath = self.checkpointpath +"/log.txt"
        self.second = 4#nnkr
        self.samplenum = self.samplingrate * self.second
        self.batch_size = 16

        #For STFT
        self.windowsize = 0.032
        self.windowsize_quant = int(self.windowsize * self.samplingrate)
        #self.windowsize_quant = 400
        self.hopsize = 0.008
        #self.hopsize_quant = int(self.hopsize * self.samplingrate)
        self.hopsize_quant = 256
        #self.n_fft = int(2**np.ceil(np.log2(self.windowsize_quant)))#フーリエ変換の係数の数
        #self.n_fft = int(2**np.ceil(np.log2(self.windowsize_quant)))#フーリエ変換の係数の数
        self.n_fft = 1024#フーリエ変換の係数の数
        self.n_mels = 80
       # self.crop_mel_frames=62#mel用の長さ
        self.crop_mel_frames=62#mel用の長さ

        #gaussianノイズを掛ける回数
        self.stepnum = 50#論文では50

        #DiffusSE Model params
        
        #self.residual_layers=30,
        self.residual_layers=30#もとは30
        self.residual_channels=64,
        self.dilation_cycle_length=10,
        #self.noise_schedule=np.linspace(1e-4, 0.035, self.stepnum+1).tolist(),
        self.noise_schedule=np.linspace(1e-4, 0.05, self.stepnum),#betaにあたる
        #self.inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.35],

        #for upsampler from specgram
        self.upsamplerconfig = {"conv1":ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8]),
                                "conv2":ConvTranspose2d(1, 1,  [3, 32], stride=[1, 16], padding=[1, 8]),
                                "output l1 size":self.samplenum}
        self.device = None
        
    def setdevice(self,device):
        self.device = device


def makeconfig(args):
    config = Myconfig(args)

    return config 