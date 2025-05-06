#!/bin/bash

#ckpt=/cs224u/duo/outputs/sentiment/2025.05.01/213454/checkpoints/best.ckpt
ckpt=/cs224u/duo/outputs/sentiment/2025.05.02/130017/checkpoints/best.ckpt

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --steps) steps="$2"; shift ;;
        --seed) seed="$2"; shift ;;
        --ckpt) ckpt="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

steps=${steps:-32}
seed=${seed:-1}
echo "  Steps: $steps"
echo "  Seed: $seed"
echo "  ckpt: $ckpt"

# Assuming the finetune_path corresponds to the DUO model
# trained for 500K steps with curriculum learning, we train the
# model for 500K more steps.
srun python -u -m main \
  mode=test_sentiment \
  seed=$seed \
  loader.batch_size=2 \
  loader.eval_batch_size=5 \
  algo=duo_base \
  data=sentiment \
  model=small \
  sampling.prompts_path=$prompts_path \
  sampling.steps=$steps \
  eval.checkpoint_path=$ckpt \
