#!/bin/zsh
#
#SBATCH --partition=p_nlp
#SBATCH --job-name=personamem_mem0
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
#SBATCH --gpus=0
#SBATCH --mem=16G
#SBATCH --array=0-3

export OPENAI_API_KEY=`cat ../api_tokens/openai_key.txt`

num_shards=4

echo "shard id: $SLURM_ARRAY_TASK_ID/$num_shards"

srun python run.py 5 mem0 gpt-4o-mini ${SLURM_ARRAY_TASK_ID} ${num_shards}
