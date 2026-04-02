#!/bin/bash

set -x

export RAY_TMPDIR=/mnt/disk1/szchen/ray_tmp
export TMPDIR=/mnt/disk1/szchen/ray_runtime
export TEMP=/mnt/disk1/szchen/ray_runtime
export TMP=/mnt/disk1/szchen/ray_runtime
export WANDB_DIR=/mnt/disk1/szchen/wandb
export WANDB_CACHE_DIR=/mnt/disk1/szchen/wandb_cache
export HF_HOME=/mnt/disk1/szchen/hf_home
export PIP_CACHE_DIR=/mnt/disk1/szchen/pip_cache
export TORCHINDUCTOR_CACHE_DIR=/mnt/disk1/szchen/torchinductor_cache
export TRITON_CACHE_DIR=/mnt/disk1/szchen/triton_cache
export XDG_CACHE_HOME=/mnt/disk1/szchen/.cache

MODEL_PATH=/mnt/disk1/weights/vlm/Qwen3-VL-8B-Instruct
CUDA_VISIBLE_DEVICES=5,7 python3 -m verl.trainer.main \
  config=examples/config.yaml \
  data.train_files=/mnt/disk1/data/rl/geometry3k/data/train-00000-of-00001.parquet \
  data.val_files=/mnt/disk1/data/rl/geometry3k/data/validation-00000-of-00001.parquet \
  worker.actor.model.model_path=${MODEL_PATH} \
  trainer.experiment_name=qwen3_vl_8b_geo_gspo_2gpu \
  trainer.n_gpus_per_node=2 \
  worker.actor.loss_type=gspo_token \
  worker.actor.loss_avg_mode=seq \
  worker.actor.clip_ratio_low=3e-4 \
  worker.actor.clip_ratio_high=4e-4 \
  worker.rollout.tensor_parallel_size=1 \
  worker.rollout.gpu_memory_utilization=0.8 \
  algorithm.disable_kl=True \
  data.rollout_batch_size=128 \
  data.val_batch_size=128

MODEL_PATH=/mnt/disk1/weights/vlm/Qwen3-VL-8B-Thinking
CUDA_VISIBLE_DEVICES=5,7 python3 -m verl.trainer.main \
  config=examples/config.yaml \
  data.train_files=/mnt/disk1/data/rl/geometry3k/data/train-00000-of-00001.parquet \
  data.val_files=/mnt/disk1/data/rl/geometry3k/data/validation-00000-of-00001.parquet \
  worker.actor.model.model_path=${MODEL_PATH} \
  trainer.experiment_name=qwen3_vl_8b_thinking_geo_gspo_2gpu \  
  trainer.n_gpus_per_node=2 \
  worker.actor.loss_type=gspo_token \
  worker.actor.loss_avg_mode=seq \
  worker.actor.clip_ratio_low=3e-4 \
  worker.actor.clip_ratio_high=4e-4 \
  worker.rollout.tensor_parallel_size=1 \
  worker.rollout.gpu_memory_utilization=0.8 \
  algorithm.disable_kl=True \
  data.rollout_batch_size=128 \
  data.val_batch_size=128