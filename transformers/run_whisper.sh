#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

# MODEL_IDs=("openai/whisper-tiny.en" "openai/whisper-small.en" "openai/whisper-base.en" "openai/whisper-medium.en" "openai/whisper-large" "openai/whisper-large-v2" "openai/whisper-large-v3" "distil-whisper/distil-medium.en" "distil-whisper/distil-large-v2" "distil-whisper/distil-large-v3" "nyrahealth/CrisperWhisper")
MODEL_IDs=("jmci/hardy-yogurt-79" "jmci/genial-shape-70" "openai/whisper-large-v3")
BATCH_SIZE=64
# Leave REVISION unset or empty for no revision
REVISION="ccb475f9be406c5f95bdfcf0fc7b386d72f00eb3"

if [ -n "$REVISION" ]; then
    REVISION_ARG="--revision=${REVISION}"
fi

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="aquavoice/cleaned_dataset_full_2x_en_resplit" \
        --dataset="default" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        ${REVISION_ARG}

    # python run_eval.py \
    #     --model_id=${MODEL_ID} \
    #     --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    #     --dataset="voxpopuli" \
    #     --split="test" \
    #     --device=0 \
    #     --batch_size=${BATCH_SIZE} \
    #     --max_eval_samples=-1 \
    #     ${REVISION_ARG}

    # python run_eval.py \
    #     --model_id=${MODEL_ID} \
    #     --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    #     --dataset="ami" \
    #     --split="test" \
    #     --device=0 \
    #     --batch_size=${BATCH_SIZE} \
    #     --max_eval_samples=-1 \
    #     ${REVISION_ARG}

    # python run_eval.py \
    #     --model_id=${MODEL_ID} \
    #     --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    #     --dataset="earnings22" \
    #     --split="test" \
    #     --device=0 \
    #     --batch_size=${BATCH_SIZE} \
    #     --max_eval_samples=-1 \
    #     ${REVISION_ARG}

    # python run_eval.py \
    #     --model_id=${MODEL_ID} \
    #     --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    #     --dataset="gigaspeech" \
    #     --split="test" \
    #     --device=0 \
    #     --batch_size=${BATCH_SIZE} \
    #     --max_eval_samples=-1 \
    #     ${REVISION_ARG}

    # python run_eval.py \
    #     --model_id=${MODEL_ID} \
    #     --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    #     --dataset="librispeech" \
    #     --split="test.clean" \
    #     --device=0 \
    #     --batch_size=${BATCH_SIZE} \
    #     --max_eval_samples=-1 \
    #     ${REVISION_ARG}

    # python run_eval.py \
    #     --model_id=${MODEL_ID} \
    #     --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    #     --dataset="librispeech" \
    #     --split="test.other" \
    #     --device=0 \
    #     --batch_size=${BATCH_SIZE} \
    #     --max_eval_samples=-1 \
    #     ${REVISION_ARG}

    # python run_eval.py \
    #     --model_id=${MODEL_ID} \
    #     --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    #     --dataset="spgispeech" \
    #     --split="test" \
    #     --device=0 \
    #     --batch_size=${BATCH_SIZE} \
    #     --max_eval_samples=-1 \
    #     ${REVISION_ARG}

    # python run_eval.py \
    #     --model_id=${MODEL_ID} \
    #     --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    #     --dataset="tedlium" \
    #     --split="test" \
    #     --device=0 \
    #     --batch_size=${BATCH_SIZE} \
    #     --max_eval_samples=-1 \
    #     ${REVISION_ARG}

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
