subsets=['train','validation','eval']
noisekinds=["bird","counstructionSite","crowd","foutain","park","rain","schoolyard","traffic","ventilation","wind_tree"]
root = "/gs/hs0/tga-shinoda/18B11396/sound-separation/data/Noise_3Voice_Disc"
for subset in subsets:
  samplelist=[]
  for noisekind in noisekinds:
    filepath="{}/{}/{}_labels.txt".format(root,noisekind,subset)
    f = open(filepath, 'r')
    datalist = f.readlines()
    for i in range(len(datalist)):
      datalist[i] = noisekind+"/"+datalist[i]
    samplelist.extend(datalist)
    
  filepath="{}/{}_alllabels.txt".format(root,subset)
  print(samplelist)
  with open(filepath, 'w') as f:
    f.writelines(samplelist)