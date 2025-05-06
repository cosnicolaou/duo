#!/bin/bash
#SBATCH -J duo-base                   # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=64000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov          # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --constraint="gpu-mid|gpu-high"
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

# To enable preemption re-loading, set `hydra.run.dir` or 
# `checkpointing.save_dir` explicitly.
finetune_path=/cs224u/cache/duo-checkpoints/duo.ckpt

# Assuming the finetune_path corresponds to the DUO model
# trained for 500K steps with curriculum learning, we train the
# model for 500K more steps.
srun python -u -m main \
  loader.batch_size=8 \
  loader.eval_batch_size=8 \
  data=sentiment \
  wandb.name=duo-owt-finetune \
  model=small \
  algo=duo_base \
  model.length=1024 \
  wandb.name=duo-base \
  training.finetune_path=$finetune_path \
  sampling.num_sample_batches=0 \
  trainer.max_steps=1000  \
  trainer.val_check_interval=1 \
  trainer.log_every_n_steps=1 \
  +wandb.offline=true