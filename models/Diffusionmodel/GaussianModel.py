import torch.nn as nn
import torch.nn.functional as F
import torch
import torchaudio
import makeconfig
import numpy as np
import gpumanager
import pprint
import wavsamplewriter
class DiffusionModel(nn.Module):    
    def __init__(self,config : makeconfig.Myconfig,Classifier,device):
        super().__init__()
        self.stepnum = config.stepnum
        self.device = device
        self.schedule_params(config)
        self.config = config
        #def STFTs
        def mystft(x):
            return torch.stft(x,
                filter_length=config.n_fft, 
                hop_length=config.hopsize_quant, 
                win_length=config.windowsize_quant,
                return_complex= False)
        def myistft(x):
            return torch.istft(x,
                filter_length=config.n_fft, 
                hop_length=config.hopsize_quant, 
                win_length=config.windowsize_quant,
                return_complex= False)
        #self.stft = mystft
        #self.istft = myistft
        #self.DiffSE = nn.parallel.DistributedDataParallel(DiffuSE(config), device_ids=[device], find_unused_parameters=True)
        self.DiffSE =DiffuSE(config).to(self.device)

    def schedule_params(self,config):
        alphas = [1]
        betas = [0]
        alphaproducts = [1.0]#総積
        alphaproduct = 1.0
        betabars = [1.0]
        betaproduct = 1.0
        noise_schedule = list(config.noise_schedule)[0]
        for i in range(self.stepnum):
            beta = noise_schedule[i]
            betas.append(beta)
            alpha = 1-beta
            alphas.append(alpha)
            alphaproduct *= alpha
            alphaproducts.append(alphaproduct)
            betaproduct *= beta
            betabars.append(betaproduct)
        #print("construct alphaprods")
        #print(alphaproducts)
        self.alphas = torch.tensor(alphas).to(self.device)# =noise_schedule
        self.betas = torch.tensor(betas).to(self.device)
        self.alphaproducts = torch.tensor(alphaproducts).to(self.device)
        self.betabars = torch.tensor(betabars).to(self.device)
        #print(betabars)
        #assertion
        for i in range(1,self.stepnum):
            diff =  (betabars[i] - betas[i]*(1-self.alphaproducts[i-1])/(1-self.alphaproducts[i]))**2
            #print("diff : {}".format(diff))
            #assert diff<0.01
        return
    
    #ノイズ除去プロセス
    def reverse(self, audio, melspec):
        audio = audio.float()
        results = [audio]
        spec = melspec
        print("len")
        print(len(self.betas))
        for t in range(self.stepnum,-1, -1):#これで最後は1で0にはならない。
            tmpalpha = 1.0/(torch.sqrt(self.alphas[t]))
            tmpalpha2 = (self.betas[t])/(torch.sqrt(1-self.alphaproducts[t]))

            gaussiannoise = torch.randn_like(audio)#xの形に合うように整形
            if t > 0:
                sigma_t = self.betas[t]*(1-self.alphaproducts[t-1])/(1-self.alphaproducts[t])
                sidma_t = torch.sqrt(sigma_t)
            else:
                #t=1
                sigma_t = self.betas[0]#=0
                print("sigma_t {}".format(sigma_t))
                print("t : {},sigma : {}".format(t,sigma_t))
            print("tmpalpha {}".format(tmpalpha2))
            tensor_t = torch.tensor([t],device=self.device)
            print(audio.size())
            print(spec.size())
            estimated  = self.estimate_gausian_noise(audio,spec,tensor_t)
            #print("estimated : ")
            #print(estimated)
            audio_tmp1 = tmpalpha.to(self.device) * (audio - tmpalpha2*estimated)
            audio_tmp2 =  sigma_t*gaussiannoise
            audio = audio_tmp1 + audio_tmp2
            audio = audio.float()
            #audio = x_(t-1)
            results.append(audio)
        stacked = torch.stack(results)
        return stacked.flip(0)#reverse反転させるプロセスなので実装上反転させる

    def get_tth_noised_signal(self,audio,t):
        gaussiannoise = torch.tensor(np.random.normal(loc=0.0, scale=1.0, size = audio.size()),device = self.device).reshape(audio.size())
        
        noised_audio = torch.sqrt(self.alphaproducts[t])*audio + torch.sqrt(1-self.alphaproducts[t])*gaussiannoise
        return noised_audio
    def get_noised_signal(self,x):
        with torch.no_grad():
            return self.get_tth_noised_signal(x,self.stepnum)

    def forward(self,audio,melspec):#x is equal to x0
        total_loss = 0
        #print("audio first shape : {}".format(audio.size()))
        melspectrogram = melspec
        arraynum = len(self.alphas)
        batch_size=audio.size()[0]
        wavelen = audio.size()[-1]
        step_array =  torch.randint(1, arraynum, size = [batch_size], device=self.device).long()
        l2norm_Coefficient = 1
        
        x_size = torch.numel(audio)#要素数
        gaussiannoise =  torch.randn_like(audio)#乱数生成
        
        noise_scales =self.alphaproducts[step_array].unsqueeze(1).unsqueeze(2).tile((1,1,wavelen))
        network_1th_input_1 = audio*torch.sqrt(noise_scales)
        network_1th_input_2 = gaussiannoise*torch.sqrt(1-noise_scales)
        #print(torch.sqrt(1-noise_scales))
        network_1th_input = network_1th_input_1 + network_1th_input_2
        network_1th_input = network_1th_input.float()
        #wavsamplewriter.wavwriter(network_1th_input[0].cpu(),"t_{}".format(step_array[0]),self.config)#チャンと数字が大きいにつれてノイズ化した
        estimated = self.estimate_gausian_noise(network_1th_input,melspectrogram,step_array)
        #tiled_gaussiannoise = gaussiannoise.unsqueeze(1).tile((1,arraynum,1,1)).flatten(0,1)
        #print("size1 : {}".format(gaussiannoise.size()))
        #print("size2 : {}".format(estimated.size()))
        #print("mean ls norm :{}".format(torch.mean((tiled_gaussiannoise - estimated)**2)))
        #total_loss = l2norm_Coefficient*l2norm/(arraynum * batch_size)
        assert estimated.size() == gaussiannoise.size()
        return estimated,gaussiannoise,step_array
    #ガウスノイズを推定する
    #output shape is equal to x
    def estimate_gausian_noise(self,x,melspectrogram,t):
        #print(t)
        #print("t shape : {}".format(t.size()))
        #t = t.unsqueeze(1)##各distributionごとにstep_arrayを割り振らせる
        #print(x.size())
        #print(melspectrogram.size())
        #print(t.size())
        return self.DiffSE((x,melspectrogram,t)) #gausiannoise

class Melconvert():    
    def __init__(self,config : makeconfig.Myconfig):
        def melspectrogram(x):
            melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate= config.samplingrate,
                win_length= config.hopsize_quant * 4,
                hop_length= config.hopsize_quant,
                n_fft= config.n_fft,
                #f_min= 20.0,
                #f_max= config.samplingrate / 2.0,
                n_mels= config.n_mels,
                power= 1.0,
                normalized= True)(x)
            #melspec = 20 * torch.log10(torch.clamp(melspec, min=1e-5)) - 20
            #melspec = torch.clamp((melspec + 100) / 100, 0.0, 1.0)
            return melspec
        self.melspec = melspectrogram
      

#these below codes are from https://github.com/neillu23/CDiffuSE/blob/main/src/cdiffuse/model.py

#Lu, Yen-Ju & Tsao, Yu & Watanabe, Shinji. (2021). A Study on Speech Enhancement Based on Diffusion Probabilistic Model.

# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from math import sqrt


Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
  layer = nn.Conv1d(*args, **kwargs)
  nn.init.kaiming_normal_(layer.weight)
  return layer


@torch.jit.script
def silu(x):
  return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
        self.projection1 = Linear(128, 512)
        self.projection2 = Linear(512, 512)

    def forward(self, diffusion_step):
        #print(self.embedding)
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]#ここ
        else:
            x = self._lerp_embedding(diffusion_step)#整数っぽい型に変換/
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        #print("steps")
        #print(steps)
        dims = torch.arange(64).unsqueeze(0)          # [1,64]
        table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
        #print("table")
        #print(table)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class SpectrogramUpsampler(nn.Module):
  def __init__(self, config,device):
    super().__init__()
    #ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)

    #original
    self.conv1 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
    self.conv2 = ConvTranspose2d(1, 1,  [3, 32], stride=[1, 16], padding=[1, 8])


    #self.conv1 = config["conv1"]
    #self.conv2 = config["conv2"]
    self.out_L = config["output l1 size"]
    self.device = device

  def forward(self, x):
    #print("spec forward")
    #print(x.size())
    x = torch.unsqueeze(x, 1)
    #print(x.size())
    x = self.conv1(x)
    #print("after conv1 {}".format(x.size()))
    x = F.leaky_relu(x, 0.4)
    x = self.conv2(x)
    #print("after conv2 {}".format(x.size()))
    outL = x.size()[-1]
    startindex = (outL-self.out_L)//2
    indices = torch.tensor(range(startindex,startindex + self.out_L ),device = self.device)
    #print(indices)
    #x = torch.index_select(x, -1, indices)
    #print(x.size())
    x = F.leaky_relu(x, 0.4)
    x = torch.squeeze(x, 1)
    return x


class ResidualBlock(nn.Module):
  def __init__(self, n_mels, residual_channels, dilation):
    super().__init__()
    self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
    self.diffusion_projection = Linear(512, residual_channels)
    self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1)
    # self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)
    self.output_projection = Conv1d(residual_channels,2* residual_channels, 1)
    #self.output_residual = Conv1d(residual_channels, residual_channels, 1)
    #self.dilated_batchnorm = nn.BatchNorm1d(128,affine=False)
    #self.conditioner_batchnorm = nn.BatchNorm1d(128,affine=False)

  def forward(self, x, conditioner, diffusion_step):
    #print("x : {}".format(x.size()))
    #print("diffusion_step : {}".format(diffusion_step.size()))
    diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
    #print(diffusion_step.size())
    #broadcast to L
    #diffusion_step = diffusion_step.tile((1,1,x.size()[-1]))
    #print(diffusion_step.size())
    conditioner = self.conditioner_projection(conditioner)
    #print("x std : {}".format(x.std()))#1くらい
    #print("x mean : {}".format(x.mean()))
    y = x + diffusion_step
    dilated = self.dilated_conv(y)
    #
    #Yuuki Nishi new
    #
    #影響をバランスするため、batchnormする
    #dilated = self.dilated_batchnorm(dilated)
    #conditioner = self.conditioner_batchnorm(conditioner)
    #print(dilated.std())
    #print(conditioner.std())
    #print("dilated std : {}".format(dilated.std()))#これがデカいのが正常らしい 4くらい
    #print("diffusion_step std : {}".format(diffusion_step.std()))#0.1くらい
    #print("conditioner std : {}".format(conditioner.std()))#0.1くらい
    #print("dilated mean : {}".format(dilated.mean()))#これがデカいのが正常らしい 4くらい
    #print("diffusion_step mean : {}".format(diffusion_step.mean()))#0.1くらい
    #print("conditioner mean : {}".format(conditioner.mean()))#0.1くらい
    
    y2 = dilated + conditioner

    #yuuki nishi end

    #y = self.dilated_conv(y)

    gate, sfilter = torch.chunk(y2, 2, dim=1)
    assert gate.size() == sfilter.size()
    y = torch.sigmoid(gate) * torch.tanh(sfilter)
    y = self.output_projection(y)
    residual, skip = torch.chunk(y, 2, dim=1)
    assert residual.size() == skip.size()
    return (x + residual) / sqrt(2.0), skip
class DiffuSE(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.residual_channels = config.residual_channels[0]
        config.residual_layers = config.residual_layers
        config.dilation_cycle_length = config.dilation_cycle_length[0]

        self.input_projection = Conv1d(1, config.residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(len(np.array(list(config.noise_schedule)[0]).flatten()) + 1)
        #↑noise_schedule にはt=0が無い
        self.spectrogram_upsampler = SpectrogramUpsampler(config.upsamplerconfig,device=config.device)
        self.residual_layers = nn.ModuleList([
            ResidualBlock(config.n_mels, config.residual_channels, 2**(i % config.dilation_cycle_length))
            for i in range(config.residual_layers)
        ])
        self.skip_projection = Conv1d(config.residual_channels, config.residual_channels, 1)
        self.output_projection = Conv1d(config.residual_channels, 1, 1)
        #self.conditioner_batchnorm = nn.BatchNorm1d(config.n_mels)
        nn.init.zeros_(self.output_projection.weight)
            
    def forward(self, input):
        audio = input[0]
        spectrogram = input[1]
        diffusion_step = input[2]
        #print("diffusion_step  : {}".format(diffusion_step.size()))
        #print("audio  : {}".format(audio.size()))
        #print("spectrogram  : {}".format(spectrogram.size()))
        #x = audio.unsqueeze(1)
        x = audio
        #print("x shape")
        #print(x.size())
        batchsize = x.size()[0]
        maybestepnum = x.size()[1]
        #print("batchsiez : {}".format(batchsize))
        #print("maybestepnum : {}".format(maybestepnum))
        #x = torch.flatten(x, start_dim=0, end_dim=1)
        x = self.input_projection(x)
        x = F.relu(x)
        # spectrogram はconditionarの為に使われる
        #print("diffusion_step shape : {}".format(diffusion_step.size()))
        #print("diffusion_step1  : {}".format(diffusion_step.size()))
        diffusion_step = self.diffusion_embedding(diffusion_step)
        #print(diffusion_step)
        #diffusion_step : [batch,512]

        #print("diffusion_step2  : {}".format(diffusion_step.size()))
        #print("diffusion_embedding shape : {}".format(diffusion_step.size()))
        #[batch*stepnum,residual_channnel]の形にする必要がある
        #diffusion_step = diffusion_step.flatten(0,1)
        #print("diffusion_step3  : {}".format(diffusion_step.size()))
        #print("after diffusion_embedding shape : {}".format(diffusion_step.size()))
        spectrogram = spectrogram.squeeze(1)
        #print("raw spec shape : {}".format(spectrogram.size()))
        spectrogram = self.spectrogram_upsampler(spectrogram)
        #spectrogram = self.conditioner_batchnorm(spectrogram)
        #print("spec shape : {}".format(spectrogram.size()))
        #spectrogram = spectrogram.unsqueeze(1).tile((1,maybestepnum,1,1)).flatten(start_dim=0, end_dim=1)
        #print("spec shape : {}".format(spectrogram.size()))
        #print("x shape : {}".format(x.size()))
        #print("diffusion_step shape : {}".format(diffusion_step.size()))
        #raw spec shape : torch.Size([3, 64, 626])
        #spec shape : torch.Size([3, 64, 80000])
        #spec shape : torch.Size([153, 64, 80000])
        # shape : torch.Size([153, 64, 80000])
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, spectrogram, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        #print("x1 shape : {}".format(x.size()))
        x = self.skip_projection(x)
        #print("x2 shape : {}".format(x.size()))
        x = F.relu(x)
        #x is [batch,channel,data length]
        x = self.output_projection(x)
        #print("x3 shape : {}".format(x.size()))
        #x is [batch,1,data length]
        #output is audio waveform
        #normalize
        #x =  self.batchnorm(x)
        return x