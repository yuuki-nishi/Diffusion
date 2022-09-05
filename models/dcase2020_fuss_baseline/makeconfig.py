from train import sounddata
configpath="/gs/hs0/tga-shinoda/18B11396/sound-separation/models/dcase2020_fuss_baseline/Config"
indpath="/gs/hs0/tga-shinoda/18B11396/sound-separation/data/Noise_Voice_Independent"
for num in [5,6,7]:
  for noisekind in sounddata.noiseclass:
    """writepath = configpath+"/{}_p{}_{}".format(noisekind,num,num)
    f = open(writepath, 'w')
    f.write("{}:1,1\n".format(noisekind))
    f.write("person:{},{}\n".format(num,num))"""

    writepath = configpath+"/p{}_{}".format(num,num)
    f = open(writepath, 'w')
    #f.write("{}:1,1\n".format(noisekind))
    f.write("person:{},{}\n".format(num,num))