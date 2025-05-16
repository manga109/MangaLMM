#!/usr/bin/sh

# MODEL_PATH=./outputs/OCR_VQA
# NUM_PARALLEL=8
MODEL_PATH=$1
NUM_PARALLEL=$2

model_name=$(basename $MODEL_PATH)
seq 0 $(($NUM_PARALLEL - 1)) | parallel --progress --jobs $NUM_PARALLEL \
    CUDA_VISIBLE_DEVICES={} python ./evaluation/inference_OCR.py \
    --model_path $MODEL_PATH \
    --output_path ./outputs/Inference_OCR \
    --split_index {} \
    --split_num $NUM_PARALLEL


# Gather results
for i in $(seq 0 $(($NUM_PARALLEL - 1))); do
    cat ./outputs/Inference_OCR/${model_name}_${i}-${NUM_PARALLEL}.jsonl >> ./outputs/Inference_OCR/${model_name}.jsonl
done

# Remove intermediate files
for i in $(seq 0 $(($NUM_PARALLEL - 1))); do
    rm ./outputs/Inference_OCR/${model_name}_${i}-${NUM_PARALLEL}.jsonl
done
