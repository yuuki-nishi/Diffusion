path=/gs/hs0/tga-shinoda/18B11396/sound-separation/models/dcase2020_fuss_baseline/train/final_model.py
python -m py_compile ${path}

files=/gs/hs0/tga-shinoda/18B11396/sound-separation/models/dcase2020_fuss_baseline/train/*.py
for file in ${files}
do

python -m py_compile ${file}

done
files=/gs/hs0/tga-shinoda/18B11396/sound-separation/models/dcase2020_fuss_baseline/*.py
for file in ${files}
do

python -m py_compile ${file}

done