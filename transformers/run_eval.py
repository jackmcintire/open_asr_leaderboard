import argparse
import os
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import AutoConfig, AutoModelForSpeechSeq2Seq, AutoModelForCTC, AutoProcessor, MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING
import evaluate
from normalizer import data_utils
import time
from tqdm import tqdm
import json
from huggingface_hub import hf_hub_download
from peft import PeftModel, PeftConfig

wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision('high')

def main(args):
    # Check if this is a PEFT model by trying to load adapter_config.json
    is_peft_model = False
    base_model_name = None
    
    try:
        # Try to download and read adapter_config.json
        adapter_config_path = hf_hub_download(
            repo_id=args.model_id, 
            filename="adapter_config.json",
            revision=args.revision
        )
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        is_peft_model = True
        base_model_name = adapter_config.get("base_model_name_or_path")
        print(f"Detected PEFT model. Base model: {base_model_name}")
    except Exception:
        # Not a PEFT model, continue with normal loading
        pass
    
    if is_peft_model and base_model_name:
        # Load the base model first
        config = AutoConfig.from_pretrained(base_model_name)
        cls_model = AutoModelForSpeechSeq2Seq if type(config) in MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING else AutoModelForCTC
        model = cls_model.from_pretrained(
            base_model_name, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="sdpa"
        )
        
        # Load and apply the PEFT adapter
        model = PeftModel.from_pretrained(
            model, 
            args.model_id,
            revision=args.revision
        )
        
        # Merge adapter weights with base model for faster inference
        model = model.merge_and_unload()
        model = model.to(args.device)
        
        # Load processor from the PEFT model repo (it should have the tokenizer files)
        processor = AutoProcessor.from_pretrained(args.model_id, revision=args.revision)
    else:
        # Original loading logic for non-PEFT models
        config = AutoConfig.from_pretrained(args.model_id, revision=args.revision)
        cls_model = AutoModelForSpeechSeq2Seq if type(config) in MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING else AutoModelForCTC
        model = cls_model.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa", revision=args.revision).to(args.device)
        processor = AutoProcessor.from_pretrained(args.model_id, revision=args.revision)
    
    model_input_name = processor.model_input_names[0]

    if model.can_generate():
        gen_kwargs = {"max_new_tokens": args.max_new_tokens}
        # getattr(model.generation_config, "is_multilingual")
        # for multilingual Whisper-checkpoints we see a definitive WER boost by setting the language and task args
        if hasattr(model.generation_config, "is_multilingual") and model.generation_config.is_multilingual:
            gen_kwargs["language"] = "en"
            gen_kwargs["task"] = "transcribe"
    elif args.max_new_tokens:
        raise ValueError("`max_new_tokens` should only be set for auto-regressive models, but got a CTC model.")

    if args.torch_compile:
        model.forward = torch.compile(model.forward, mode=args.compile_mode, fullgraph=True)
        if model.can_generate():
            # enable static k/v cache for autoregressive models
            model.generation_config.cache_implementation = "static"

    def benchmark(batch, min_new_tokens=None):
        # Load audio inputs
        audios = [audio["array"] for audio in batch["audio"]]
        minibatch_size = len(audios)

        # START TIMING
        start_time = time.time()

        # 1. Pre-Processing
        # 1.1 Pad audios to max batch size if using torch compile to prevent re-compilations
        padding_size = None
        if minibatch_size != args.batch_size and args.torch_compile:
            padding_size = args.batch_size - minibatch_size
            padding_audios = [audios[-1] for _ in range(padding_size)]
            audios.extend(padding_audios)

        if not model.can_generate(): #or len(audios[0]) > processor.feature_extractor.n_samples:
            # 1.2 Either CTC pre-processing (normalize to mean 0, std 1), or long-form Whisper processing
            inputs = processor(
                audios,
                sampling_rate=16_000,
                truncation=False,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            )
        else:
            # 1.3 Standard Whisper processing: pad audios to 30-seconds and converted to log-mel
            inputs = processor(audios, sampling_rate=16_000, return_tensors="pt", device=args.device)

        inputs = inputs.to(args.device)
        inputs[model_input_name] = inputs[model_input_name].to(torch.bfloat16)

        # 2. Model Inference
        with sdpa_kernel(SDPBackend.MATH if args.torch_compile else SDPBackend.FLASH_ATTENTION):
            if model.can_generate():
                # 2.1 Auto-regressive generation for encoder-decoder models
                pred_ids = model.generate(**inputs, **gen_kwargs, min_new_tokens=min_new_tokens)
            else:
                # 2.2. Single forward pass for CTC
                with torch.no_grad():
                    logits = model(**inputs).logits
                    pred_ids = logits.argmax(-1)

        # 3. Post-processing
        # 3.1 Strip padded ids from predictions
        if padding_size is not None:
            pred_ids = pred_ids[:-padding_size, ...]

        # 3.2 Convert token ids to text transcription
        pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)

        # END TIMING
        runtime = time.time() - start_time

        # normalize by minibatch size since we want the per-sample time
        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]

        # normalize transcriptions with English normalizer
        batch["predictions"] = [data_utils.normalizer(pred) for pred in pred_text]
        batch["references"] = batch["norm_text"]
        return batch

    if args.warmup_steps is not None:
        dataset = data_utils.load_data(args)
        dataset = data_utils.prepare_data(dataset)

        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = dataset.take(num_warmup_samples)
        else:
            warmup_dataset = dataset.select(range(min(num_warmup_samples, len(dataset))))
        warmup_dataset = iter(warmup_dataset.map(benchmark, batch_size=args.batch_size, batched=True, fn_kwargs={"min_new_tokens": args.max_new_tokens}))

        for _ in tqdm(warmup_dataset, desc="Warming up..."):
            continue

    dataset = data_utils.load_data(args)
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))
    dataset = data_utils.prepare_data(dataset)

    dataset = dataset.map(
        benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"],
    )

    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
    }
    result_iter = iter(dataset)
    for result in tqdm(result_iter, desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])

    # Write manifest results (WER and RTFX)
    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    wer = wer_metric.compute(
        references=all_results["references"], predictions=all_results["predictions"]
    )
    wer = round(100 * wer, 2)
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
    print("WER:", wer, "%", "RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with 🤗 Transformers",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="esb/datasets",
        help="Dataset path. By default, it is `esb/datasets`",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name. *E.g.* `'librispeech_asr` for the LibriSpeech ASR dataset, or `'common_voice'` for Common Voice. The full list of dataset names "
        "can be found at `https://huggingface.co/datasets/esb/datasets`",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset. *E.g.* `'validation`' for the dev split, or `'test'` for the test split.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate (for auto-regressive models).",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Whether to JIT compile the forward pass of the model.",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="max-autotune",
        help="Mode for torch compiling model forward pass. Can be either 'default', 'reduce-overhead', 'max-autotune' or 'max-autotune-no-cudagraphs'.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision (commit hash, branch name, or tag name) to use. If not provided, uses the latest version.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="Number of warm-up steps to run before launching the timed runs.",
    )
    args = parser.parse_args()
    parser.set_defaults(streaming=False)

    main(args)
