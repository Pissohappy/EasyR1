#!/bin/bash

set -x

export RAY_TMPDIR=/mnt/disk1/szchen/ray_tmp
export TMPDIR=/mnt/disk1/szchen/ray_runtime
export TEMP=/mnt/disk1/szchen/ray_runtime
export TMP=/mnt/disk1/szchen/ray_runtime

MODEL_PATH=/mnt/disk1/weights/vlm/Qwen3-VL-8B-Instruct  # replace it with your local file path
CUDA_VISIBLE_DEVICES=4,5 python3 -m verl.trainer.main \
  config=examples/config.yaml \
  data.train_files=/mnt/disk1/data/rl/geometry3k/data/train-00000-of-00001.parquet \
  data.val_files=/mnt/disk1/data/rl/geometry3k/data/validation-00000-of-00001.parquet \
  worker.actor.model.model_path=${MODEL_PATH} \
  trainer.experiment_name=qwen3_vl_8b_geo_grpo_3gpu \
  trainer.n_gpus_per_node=2 \
  worker.rollout.tensor_parallel_size=1 \
  worker.rollout.gpu_memory_utilization=0.4 \
  data.rollout_batch_size=128 \
  data.val_batch_size=128