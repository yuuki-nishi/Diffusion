import torchaudio
import makeconfig
import os
import torch
import numpy as np
def wavwriter(data,name,config : makeconfig.Myconfig):
	
    if not os.path.exists(config.wavdirpath):
        os.makedirs(config.wavdirpath)
    #if data.ndim() <= 1:
    #    data = data.unsqueeze(0)
    path = config.wavdirpath + "/{}.wav".format(name)

    print("path : {}".format(path))
    data = np.expand_dims(data,0)
    data = torch.from_numpy(data)
    data=data/torch.max(torch.abs(data))#ファイルに書き込むときはこうする(計算上は絶対値が1を超ええる)
    torchaudio.save(filepath=path, src=data, sample_rate=config.samplingrate)

    return