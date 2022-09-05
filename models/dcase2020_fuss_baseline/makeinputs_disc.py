noisekinds = ["bird","counstructionSite","crowd","foutain","park","rain","schoolyard","traffic","ventilation","wind_tree"]
nums = [4,5,6]
cpath="/gs/hs0/tga-shinoda/18B11396/sound-separation/models/dcase2020_fuss_baseline/DiscConfig"

for kind in noisekinds:
  for num in nums:
    """filename = "./inputparams_disc/{}_p{}.sh".format(kind,num)
    f = open(filename,"w")
    f.write("VoiceExec_name={}_p{}\n".format(kind,num))
    #f.write("VoiceExec_name=p{}\n".format(num))
    name= "{}_p{}".format(kind,num)
    f.write("ConfigPath={}/{}\n".format(cpath,name))"""

    filename = "./inputparams_disc/p{}.sh".format(num)
    f = open(filename,"w")
    f.write("VoiceExec_name=p{}\n".format(num))
    #f.write("VoiceExec_name=p{}\n".format(num))
    name= "p{}".format(num)
    f.write("ConfigPath={}/{}\n".format(cpath,name))