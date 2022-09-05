noisekinds = ["bird","counstructionSite","crowd","foutain","park","rain","schoolyard","traffic","ventilation","wind_tree"]
nums = [5,6,7]
cpath="/gs/hs0/tga-shinoda/18B11396/sound-separation/models/dcase2020_fuss_baseline/Config"
for kind in noisekinds:
  for num in nums:
    """filename = "./inputparams/{}_p{}_{}.sh".format(kind,num,num)
    f = open(filename,"w")
    f.write("VoiceExec_name={}_p{}_{}\n".format(kind,num,num))
    f.write("OutNum={}\n".format(num+1))
    f.write("NoiseKind={}\n".format(kind))
    name= "{}_p{}_{}".format(kind,num,num)
    f.write("ConfigPath={}/{}\n".format(cpath,name))"""
    #pのみ
    filename = "./inputparams/p{}_{}.sh".format(num,num)
    f = open(filename,"w")
    f.write("VoiceExec_name=p{}_{}\n".format(num,num))
    f.write("OutNum={}\n".format(num))
    #f.write("NoiseKind={}\n".format(kind))
    name= "p{}_{}".format(num,num)
    f.write("ConfigPath={}/{}\n".format(cpath,name))