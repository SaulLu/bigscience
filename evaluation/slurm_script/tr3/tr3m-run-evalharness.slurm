#!/bin/bash
#SBATCH --job-name=tr7a-1B3-alibi-eval
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=10       # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:1                # number of gpus
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsdswork/projects/rech/six/uue59kq/logs/%x-%j.out          # output file name
#SBATCH --error=/gpfsdswork/projects/rech/six/uue59kq/logs/%x-%j.err           # error file name
#SBATCH --account=six@gpu

set -x -e

DATA_OUTPUT_PATH=$ALL_CCFRSCRATCH/synched_exps/tr3m-1B3-pile/checkpoints
CHECKPOINT_PATH=$DATA_OUTPUT_PATH/global_step117000/
MEGATRON_DEEPSPEED_REPO=$SCRATCH/repos/Megatron-DeepSpeed

REPO_PATH=$SCRATCH/synched_exps/tr3m-1B3-pile/tr3m-1B3-pile-logs
LOGS_PATH=$REPO_PATH/logs
RESULTS_PATH=$REPO_PATH/eval/results-eval-harness.json

mkdir -p $(dirname $RESULTS_PATH)
mkdir -p $LOGS_PATH

VOCAB_FILE=$MEGATRON_DEEPSPEED_REPO/data/gpt2-vocab.json
MERGE_FILE=$MEGATRON_DEEPSPEED_REPO/data/gpt2-merges.txt

# defining the right environment variables
source $six_ALL_CCFRWORK/start-prod
conda activate lucile-alibi
export TRANSFORMERS_CACHE=$six_ALL_CCFRWORK/models
export HF_DATASETS_CACHE=$six_ALL_CCFRWORK/datasets
export HF_MODULES_CACHE=$six_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$six_ALL_CCFRWORK/metrics
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd $MEGATRON_DEEPSPEED_REPO/tasks/eval_harness

PP_SIZE=1  
TP_SIZE=1  
N_GPUS=1

#dummy arguments to make megatron happy.
MEGATRON_REQUIRED_ARGS="\
    --num-layers -1\
    --hidden-size -1\
    --num-attention-heads -1\
    --seq-length -1 \
    --max-position-embeddings -1
"

CMD="evaluate.py \
    --load $CHECKPOINT_PATH\
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE\
    --vocab-file $VOCAB_FILE\
    --merge-file $MERGE_FILE\
    --micro-batch-size 64\
    --adaptive_seq_len\
    --eval_fp32\
    --task_list hellaswag,mrpc,piqa\
    --results_path $RESULTS_PATH\
    $MEGATRON_REQUIRED_ARGS\
    "

LAUNCHER="deepspeed --num_gpus $N_GPUS"

echo $CMD

# to debug - add echo (it exits and prints what it would have launched)
$LAUNCHER $CMD 2>&1 | tee -a $LOGS_PATH/tr3m-1B3-pile.$SLURM_JOBID.out