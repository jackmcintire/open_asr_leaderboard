#!/bin/bash

# Benchmarking script for Voxtral-Mini-3B-2507 model
# Note: This script is experimental and may require adjustments

export PYTHONPATH="..":$PYTHONPATH

MODEL_IDs=("mistralai/Voxtral-Mini-3B-2507")
BATCH_SIZE=8  # Start with small batch size due to model size (4.68B params)
# Leave REVISION unset or empty for no revision
REVISION=""

if [ -n "$REVISION" ]; then
    REVISION_ARG="--revision=${REVISION}"
fi

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    # python run_voxtral.py \
    #     --model_id=${MODEL_ID} \
    #     --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    #     --dataset="tedlium" \
    #     --split="test" \
    #     --device=0 \
    #     --batch_size=${BATCH_SIZE} \
    #     --max_eval_samples=-1 \
    #     ${REVISION_ARG}
    
    # python run_voxtral.py \
    #     --model_id=${MODEL_ID} \
    #     --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    #     --dataset="librispeech" \
    #     --split="test.clean" \
    #     --device=0 \
    #     --batch_size=${BATCH_SIZE} \
    #     --max_eval_samples=-1 \
    #     ${REVISION_ARG}
    
    # python run_voxtral.py \
    #     --model_id=${MODEL_ID} \
    #     --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    #     --dataset="librispeech" \
    #     --split="test.other" \
    #     --device=0 \
    #     --batch_size=${BATCH_SIZE} \
    #     --max_eval_samples=-1 \
    #     ${REVISION_ARG}
    
    # python run_voxtral.py \
    #     --model_id=${MODEL_ID} \
    #     --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    #     --dataset="voxpopuli" \
    #     --split="test" \
    #     --device=0 \
    #     --batch_size=${BATCH_SIZE} \
    #     --max_eval_samples=-1 \
    #     ${REVISION_ARG}
    # # AMI
    # python run_voxtral.py \
    #     --model_id=${MODEL_ID} \
    #     --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    #     --dataset="ami" \
    #     --split="test" \
    #     --device=0 \
    #     --batch_size=${BATCH_SIZE} \
    #     --max_eval_samples=-1 \
    #     ${REVISION_ARG}
    
    # python run_voxtral.py \
    #     --model_id=${MODEL_ID} \
    #     --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    #     --dataset="earnings22" \
    #     --split="test" \
    #     --device=0 \
    #     --batch_size=${BATCH_SIZE} \
    #     --max_eval_samples=-1 \
    #     ${REVISION_ARG}
    
    python run_voxtral.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="gigaspeech" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        ${REVISION_ARG}
    
    python run_voxtral.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="spgispeech" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        ${REVISION_ARG}
    
    python run_voxtral.py \
        --model_id=${MODEL_ID} \
        --dataset_path="aquavoice/cleaned_dataset_full_2x_en_resplit" \
        --dataset="default" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        ${REVISION_ARG}
    
    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done