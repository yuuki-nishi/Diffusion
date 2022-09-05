from pydub import AudioSegment
#sound = AudioSegment.from_wav("/gs/hs0/tga-shinoda/18B11396/sound-separation/data/Isolated_urban_sound_database/background/bird_bg01.wav")
#sound = sound.set_channels(1)
#sound.export("tmp.wav", format="wav")
#import soundfile
import glob
import os
'''
files = glob.glob("../../data/Isolated_urban_sound_database/background/*wav", recursive=True)
f = open('list.txt', 'w')
f.write("background\n")
for wavfile in files:
  # 音声ファイルの読み込み
  sound = AudioSegment.from_file(wavfile, "wav")

  # 情報の取得
  time = sound.duration_seconds # 再生時間(秒)
  name = os.path.basename(wavfile)

  # 情報の表示
  f.write("Time : {:.2e}second, Name: {}\n".format(time,name))
files = glob.glob("../../data/Isolated_urban_sound_database/event/*wav", recursive=True)
#f = open('list.txt', 'w')
f.write("event\n")
for wavfile in files:
  # 音声ファイルの読み込み
  sound = AudioSegment.from_file(wavfile, "wav")

  # 情報の取得
  time = sound.duration_seconds # 再生時間(秒)
  name = os.path.basename(wavfile)

  # 情報の表示
  f.write("Time : {:.2e}second, Name: {}\n".format(time,name))
'''
#stereo2monoral
files = glob.glob("../../data/Isolated_urban_sound_database/background/*wav", recursive=True)
for wavfile in files:
  sound = AudioSegment.from_wav(wavfile)
  sound = sound.set_channels(1)
  
  name = os.path.basename(wavfile)
  sound.export("/gs/hs0/tga-shinoda/18B11396/sound-separation/data/Monoral_Isolated_urban_sound/background/{}".format(name), format="wav")
