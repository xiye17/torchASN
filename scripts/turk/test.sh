#!/bin/bash
set -e

test_file="data/turk/test.bin"
model_name=$(basename $1)
max_decode_step=70
beam_size=10

python -u test.py \
    --test_file ${test_file} \
    --model_file $1 \
    --beam_size ${beam_size} \
    --max_decode_step 70

