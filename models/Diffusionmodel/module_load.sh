#. /etc/profile.d/modules.sh
module avail
module load python #pytorchやるときはこれやっちゃダメ moduleの方使っちゃう?
module load cuda
module load python-extensions
#module load intel
module load cudnn
module load pytorch
module list