SPEECH_DATA_PATH=/net/shard/work/y-nishi/dataset/LJSpeech-1.1/wavs
#TRAIN_SPLIT=si_et_20:si_dt_20:sd_et_20
TRAIN_SPLIT=
EVAL_SPLIT=si_dt_05/22g

NOISE_DATA_PATH=../data/Voice_Splited

EXEC_ROOT=/net/shard/work/y-nishi/SoundSeparation/models/Diffusionmodel

#export PATH=/home/9/18B11396/miniconda3/bin/python3
#export PATH=/home/9/18B11396/.local/bin
#pip3 cache purge
#wget https://bootstrap.pypa.io/pip/3.4/get-pip.py
#python3 -user get-pip.py
#python3 -m pip install --user  --upgrade pip
echo "tmp" >&2
#インストール先を指定
#python3 -m pip install torch torchvision torchaudio torchmetrics torch-summary -t /gs/hs0/tga-shinoda/18B11396/sound-separation/lib