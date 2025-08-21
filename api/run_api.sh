#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

export OPENAI_API_KEY="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"
export ASSEMBLYAI_API_KEY="your_api_key"
export ELEVENLABS_API_KEY="your_api_key"
export REVAI_API_KEY="your_api_key"

MODEL_IDs=(
    "openai/gpt-4o-transcribe"
    # "openai/gpt-4o-mini-transcribe"
    # "openai/whisper-1"
    # "assembly/best"
    # "elevenlabs/scribe_v1"
    # "revai/machine" # please use --use_url=True
    # "revai/fusion" # please use --use_url=True
    # "speechmatics/enhanced"
    # "avalon-b200-dev"
)

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}
    python run_eval.py \
        --dataset_path="jmci/aispeak-v1" \
        --dataset="default" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers=100

    python run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="ami" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers=100

    python run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="earnings22" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers=100

    python run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="gigaspeech" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers=100

    python run_eval.py \
        --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
        --dataset "librispeech" \
        --split "test.clean" \
        --model_name ${MODEL_ID} \
        --max_workers=100

    python run_eval.py \
        --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
        --dataset "librispeech" \
        --split "test.other" \
        --model_name ${MODEL_ID} \
        --max_workers=100

    python run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="spgispeech" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers=100

    python run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="tedlium" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers=100

    python run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="voxpopuli" \
        --split="test" \
        --model_name ${MODEL_ID} \
        --max_workers=100
    
    # python run_eval.py \
    #     --dataset_path="aquavoice/cleaned_dataset_full_2x_en_resplit" \
    #     --dataset="default" \
    #     --split="test" \
    #     --model_name ${MODEL_ID} \
    #     --max_workers=100
    
    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
