#!/bin/bash

#SBATCH --gres=gpu:1
@@ACCOUNT@@
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --job-name=@@JOB_NAME@@
#SBATCH --mem=40000
#SBATCH --time=@@JOB_TIME@@
#SBATCH --output=@@JOB_LOG@@
#SBATCH --partition=@@PARTITION@@
@@QOS@@
. /usr/local/anaconda/5.1.0-Python3.6-gcc5/etc/profile.d/conda.sh
conda activate @@CONDA_ENV@@

echo "run job script"


echo "Training the model ..."
python3 @@TRAIN@@ -gpu_ranks 0 -config @@CONFIG@@ > @@TRAING_LOG@@ 2>&1

echo "Training is finished!"

bash @@TEST_SCRIPT@@