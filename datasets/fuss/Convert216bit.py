import soundfile
import glob
#なぜか32bitなので16bitに変換
files = glob.glob("../../data/fuss_augment_2020/Voice_max4_Named_Max3/**/*wav", recursive=True)
for wavfile in files:
  data, samplerate = soundfile.read(wavfile)
  #print(data)
  soundfile.write(wavfile, data, samplerate, subtype='PCM_16')
