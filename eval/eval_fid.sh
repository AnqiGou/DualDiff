#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 
/python -m pytorch_fid \
    ./eval/FID/laion-40k \
    ./generate_images