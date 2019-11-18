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

BPE_REMOVER=@@BPE_REMOVER@@
TRANSLATE=@@TRANSLATE@@
TEST_SRC=@@TEST_SRC@@
TEST_TGT=@@TEST_TGT@@
DEV_SRC=@@DEV_SRC@@
DEV_TGT=@@DEV_TGT@@
BLEU=@@BLEU@@
TEST_TGT_RAW=@@TEST_TGT_RAW@@
DEV_TGT_RAW=@@DEV_TGT_RAW@@

. /usr/local/anaconda/5.1.0-Python3.6-gcc5/etc/profile.d/conda.sh
conda activate gradPytorch

BASEDIR=$(dirname "$0")

echo "Testing model(s) ..."
for model in $BASEDIR/checkpoints/model_T0_*.pt
do
         echo ${model}
        python3 $TRANSLATE -gpu 0 -model $model -report_bleu -output ${model}.preds.bpe -src $TEST_SRC -tgt $TEST_TGT > ${model}.stats 2>&1
        python3 $TRANSLATE -gpu 0 -model $model -report_bleu -output ${model}.dpreds.bpe -src $DEV_SRC -tgt $DEV_TGT > ${model}.dstats 2>&1

         python $BPE_REMOVER ${model}.preds.bpe ${model}.preds
         python $BPE_REMOVER ${model}.dpreds.bpe ${model}.dpreds

         perl $BLEU $TEST_TGT_RAW < ${model}.preds > ${model}.tbleu
         perl $BLEU $DEV_TGT_RAW < ${model}.dpreds > ${model}.dbleu
done

echo "Testing is finished!"