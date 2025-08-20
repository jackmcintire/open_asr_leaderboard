#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

# MODEL_IDs=("openai/whisper-tiny.en" "openai/whisper-small.en" "openai/whisper-base.en" "openai/whisper-medium.en" "openai/whisper-large" "openai/whisper-large-v2" "openai/whisper-large-v3" "distil-whisper/distil-medium.en" "distil-whisper/distil-large-v2" "distil-whisper/distil-large-v3" "nyrahealth/CrisperWhisper")
MODEL_IDs=("openai/whisper-large-v3" "aquavoice/sweet-lion-95")
BATCH_SIZE=128
# Leave REVISION unset or empty for no revision
REVISION=""

if [ -n "$REVISION" ]; then
    REVISION_ARG="--revision=${REVISION}"
fi

PROMPT="Jack McIntire, Speechmatics, process.env, bun run dev, Clausewitz"
num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="voxpopuli" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --prompt="${PROMPT}" \
        ${REVISION_ARG}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="ami" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --prompt="${PROMPT}" \
        ${REVISION_ARG}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="earnings22" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --prompt="${PROMPT}" \
        ${REVISION_ARG}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="gigaspeech" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --prompt="${PROMPT}" \
        ${REVISION_ARG}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.clean" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --prompt="${PROMPT}" \
        ${REVISION_ARG}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.other" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --prompt="${PROMPT}" \
        ${REVISION_ARG}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="spgispeech" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        ${REVISION_ARG}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="tedlium" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --prompt="${PROMPT}" \
        ${REVISION_ARG}
    
    # python run_eval.py \
    #     --model_id=${MODEL_ID} \
    #     --dataset_path="aquavoice/cleaned_dataset_full_2x_en_resplit" \
    #     --dataset="default" \
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
