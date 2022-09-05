path="/gs/hs0/tga-shinoda/18B11396/sound-separation/data/Isolated_urban_sound_database/background"
path2="/gs/hs0/tga-shinoda/18B11396/sound-separation/data/Monoral_Isolated_urban_sound/background2"
persons=["001","002","400"]
noises=["bird","constructionSite","crowd","fountain","park","rain","schoolyard","traffic","ventilation","wind_tree"]
import glob
import os
import wave
import numpy as np
split=["train","eval","validation"]
for p in noises:
  tmppath=path2+"/"+p+"_bg/**/*.wav"
  files=glob.glob(tmppath)
  lens = []
  for file in files:
    label=os.path.basename(file).split("_")[0]
    if label == p or label + "_tree"== p :
      # 読み込みモードでWAVファイルを開く
      with wave.open(file,  'rb') as wr:
      
          # 情報取得
          ch = wr.getnchannels()
          width = wr.getsampwidth()
          fr = wr.getframerate()
          fn = wr.getnframes()
      
          # 表示
          #print("チャンネル: ", ch)
          #print("サンプルサイズ: ", width)
          #print("サンプリングレート: ", fr)
          #print("フレームレート: ", fn)
          #rint("再生時間: ", 1.0 * fn / fr)
          lens.append( 1.0 * fn / fr)

  print("{0} & {1} & {2:.2f} & {3:.2f} \\\\".format(p , int(np.sum(lens)), np.mean(lens),np.std(lens)))
