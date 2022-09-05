#! /bin/sh
#$ -cwd
#$ -l s_core=1
#it was 24:00:00
#$ -l h_rt=1:00:00
#$ -N makeconfig
#$ -o TsubameOutputs/makeconfig/tmp_e
#$ -e TsubameOutputs/makeconfig/tmp_o
. /etc/profile.d/modules.sh

python3 makeconfig.py
python3 makeconfig_disc.py
python3 makeinputparams.py
python3 makeinputs_disc.py