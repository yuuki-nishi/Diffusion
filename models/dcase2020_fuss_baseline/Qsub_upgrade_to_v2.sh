#! /bin/sh
#$ -cwd
#$ -l q_node=1
#it was 24:00:00
#$ -l h_rt=1:0:00
#$ -N test
#$ -o TsubameOutputs/upgrade_log/tmp_e
#$ -e TsubameOutputs/upgrade_log/tmp_o
. /etc/profile.d/modules.sh
# upgrade codes to tensorflow v2
module load cuda/11.0.194
module load nccl/2.8.4
#module load intel
module load cudnn/8.1
module load python/3.6.5
module load tensorflow
tf_upgrade_v2 \
  --infile train/network.py \
  --outfile train/network.py \
  --reportfile report.txt