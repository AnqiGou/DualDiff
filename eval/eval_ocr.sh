#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
/python eval/eval_dgocr.py \
        --img_dir ./generate_images \
        --input_json ./data//laion_word/test1k.json