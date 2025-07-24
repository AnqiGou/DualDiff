#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python eval/eval_clip.py \
        --img_dir ./generate_images \
        --input_json ./data/benchmark/laion_word/test1k.json