#!/bin/bash
#SBATCH -J an_owt_duo                    # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=16000                   # server memory requested (per node)
#SBATCH -t 24:00:00                   # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov,gpu      # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

export HYDRA_FULL_ERROR=1

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --steps) steps="$2"; shift ;;
        --seed) seed="$2"; shift ;;
        --ckpt) ckpt="$2"; shift ;;
        --prompts_path) prompts_path="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

checkpoint_path=/cs224u/cache/duo-checkpoints


steps=${steps:-32}
seed=${seed:-1}
prompts_path=${prompts_path:-"/cs224u/duo/prompts.txt"}
echo "  Steps: $steps"
echo "  Seed: $seed"
echo "  ckpt: $ckpt"
echo "  prompts_path: $prompts_path"

srun python -u -m main \
  mode=prompt \
  seed=$seed \
  loader.batch_size=2 \
  loader.eval_batch_size=2 \
  algo=duo_base \
  model=small \
  sampling.prompts_path=$prompts_path \
  sampling.steps=$steps \
  eval.checkpoint_path=$checkpoint_path/$ckpt.ckpt \
  sampling.noise_removal=greedy \
  eval.generated_samples_path=$checkpoint_path/samples_ancestral/$seed-$steps-ckpt-$ckpt-greedy-prompt.json \
