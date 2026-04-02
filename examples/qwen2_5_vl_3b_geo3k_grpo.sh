#!/bin/bash

set -x

MODEL_PATH=/mnt/disk1/weights/vlm/Qwen3-VL-8B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/mnt/disk1/data/rl/geometry3k@train \
    data.val_files=/mnt/disk1/data/rl/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen3_vl_3b_geo_grpo \
    trainer.n_gpus_per_node=2
