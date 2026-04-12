#!/usr/bin/env bash

python griffey/train.py \
  --batch_size 32 \
  --warmup_epochs 3 \
  --finetune_epochs 20 \
  --checkpoint_dir ./checkpoints
