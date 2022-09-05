#! /bin/sh
#$ -cwd
#$ -l q_node=1
#it was 24:00:00
#$ -l h_rt=24:00:00
#$ -N fuss_tr
. /etc/profile.d/modules.sh

#git add .
git commit -m "init repository"