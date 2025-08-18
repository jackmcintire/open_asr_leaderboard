#!/usr/bin/env python3
import argparse
import os
import time
from typing import List

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

from transformers import VoxtralForConditionalGeneration, AutoProcessor
import evaluate

# ESB helpers (repo-local)
from normalizer import data_utils

wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision("high")


def _device_str(device_index: int) -> str:
    if device_index >= 0 and torch.cuda.is_available():
        return f"cuda:{device_index}"
    return "cpu"


def _infer_formats(audio_batch: List[dict]) -> List[str]:
    """
    VoxtralProcessor requires `format` when audio is provided as arrays/tensors.
    We conservatively return 'wav' for each sample. If paths/extensions are present,
    we try to use them; otherwise we fallback to 'wav'.
    """
    formats = []
    for a in audio_batch:
        fmt = None
        # If a path is present, try to derive from extension
        path = a.get("path") if isinstance(a, dict) else None
        if path:
            ext = os.path.splitext(path)[1].lower().lstrip(".")
            if ext in {"wav", "mp3", "flac", "ogg", "m4a", "webm", "aac"}:
                fmt = ext
        # Final fallback
        formats.append(fmt or "wav")
    return formats


def benchmark_voxtral(batch, *, processor, model, args, device_str, dtype):
    """Transcribe one (batched) chunk and return metrics + hyps/refs."""
    # 1) Gather audio arrays + lengths
    audios = [el["array"] for el in batch["audio"]]
    srs = [int(el["sampling_rate"]) for el in batch["audio"]]
    # Use the first SR; your data_utils typically resamples to a fixed rate (e.g., 16k)
    sampling_rate = srs[0]
    # Defensive: if SRs differ, let processor handle; but warn
    if any(sr != sampling_rate for sr in srs):
        print(f"[warn] Mixed sampling rates in batch: {srs}. Using first={sampling_rate}.")

    batch["audio_length_s"] = [len(el["array"]) / el["sampling_rate"] for el in batch["audio"]]

    # 2) Build Voxtral transcription request
    #    IMPORTANT: Provide `format` when passing arrays/tensors.
    formats = _infer_formats(batch["audio"])

    start_time = time.time()
    inputs = processor.apply_transcription_request(
        language=args.language,               # e.g. "en"
        audio=audios,                         # list[np.ndarray]
        model_id=args.model_id,               # hf repo id
        sampling_rate=sampling_rate,
        format=formats,
    )
    inputs = inputs.to(device_str, dtype=dtype)

    # 3) Generate
    gen_kwargs = {}
    if args.max_new_tokens is not None and args.max_new_tokens > 0:
        gen_kwargs["max_new_tokens"] = args.max_new_tokens
    # For transcription, Mistral recommends temperature=0.0; expose anyway
    gen_kwargs["temperature"] = args.temperature
    gen_kwargs["do_sample"] = args.temperature > 0.0

    with sdpa_kernel(SDPBackend.MATH if args.torch_compile else SDPBackend.FLASH_ATTENTION):
        if args.torch_compile and not getattr(model, "_compiled", False):
            # fullgraph=False handles variable-length audio better
            model.forward = torch.compile(model.forward, mode=args.compile_mode, fullgraph=False)
            model._compiled = True
        outputs = model.generate(**inputs, **gen_kwargs)

    # 4) Decode ONLY newly generated tokens after the prompt
    prompt_len = inputs.input_ids.shape[1]
    preds = processor.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)

    runtime = time.time() - start_time
    bs = len(audios)
    batch["transcription_time_s"] = [runtime / bs] * bs

    # 5) Normalize + references
    refs = batch["norm_text"] if "norm_text" in batch else batch.get("text", [""] * bs)
    batch["predictions"] = [data_utils.normalizer(p) for p in preds]
    batch["references"] = refs
    return batch


def main(args):
    device_str = _device_str(args.device)
    dtype = torch.bfloat16 if device_str.startswith("cuda") else torch.float32

    # Load Voxtral
    processor = AutoProcessor.from_pretrained(args.model_id, revision=args.revision)
    model = VoxtralForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=dtype if device_str.startswith("cuda") else torch.float32,
        revision=args.revision
    ).to(device_str)

    # Warmup (optional)
    if args.warmup_steps is not None and args.warmup_steps > 0:
        dataset_w = data_utils.prepare_data(data_utils.load_data(args))
        num_warmup = args.warmup_steps * args.batch_size
        warm = dataset_w.take(num_warmup) if args.streaming else dataset_w.select(range(min(num_warmup, len(dataset_w))))
        _ = warm.map(
            benchmark_voxtral,
            batch_size=args.batch_size,
            batched=True,
            fn_kwargs={"processor": processor, "model": model, "args": args, "device_str": device_str, "dtype": dtype},
        )

    # Timed evaluation
    dataset = data_utils.prepare_data(data_utils.load_data(args))

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        dataset = dataset.take(args.max_eval_samples) if args.streaming else dataset.select(
            range(min(args.max_eval_samples, len(dataset)))
        )

    dataset = dataset.map(
        benchmark_voxtral,
        batch_size=args.batch_size,
        batched=True,
        remove_columns=["audio"],
        fn_kwargs={"processor": processor, "model": model, "args": args, "device_str": device_str, "dtype": dtype},
    )

    # Collect results
    all_results = {"audio_length_s": [], "transcription_time_s": [], "predictions": [], "references": []}
    for row in dataset:
        for k in all_results:
            all_results[k].append(row[k])

    # Write manifest + report metrics
    manifest_path = data_utils.write_manifest(
        all_results["references"], all_results["predictions"],
        args.model_id, args.dataset_path, args.dataset, args.split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    wer = wer_metric.compute(references=all_results["references"], predictions=all_results["predictions"])
    wer = round(100 * wer, 2)
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
    print("WER:", wer, "%", "RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, required=True,
                        help="Hugging Face model id, e.g., 'mistralai/Voxtral-Mini-3B-2507'")
    parser.add_argument("--dataset_path", type=str, default="esb/datasets",
                        help="Dataset path (e.g., 'hf-audio/esb-datasets-test-only-sorted').")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name in ESB loader (e.g., 'librispeech').")
    parser.add_argument("--split", type=str, default="test",
                        help="Split name (e.g., 'test.clean', 'test.other', etc.).")
    parser.add_argument("--device", type=int, default=-1,
                        help="Device index: -1 for CPU, 0 for first GPU, etc.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for dataset.map (start small; Voxtral is large).")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Limit number of samples; set >0 for a quick smoke test.")
    parser.add_argument("--no-streaming", dest="streaming", action="store_false",
                        help="If set, load dataset into memory (non-streaming).")
    parser.add_argument("--max_new_tokens", type=int, default=4000,
                        help="Max tokens to generate per sample (transcripts can be long).")
    parser.add_argument("--torch_compile", action="store_true",
                        help="Compile model.forward (uses fullgraph=False internally).")
    parser.add_argument("--compile_mode", type=str, default="max-autotune",
                        help="Torch compile mode: 'default', 'reduce-overhead', 'max-autotune', ...")
    parser.add_argument("--revision", type=str, default=None,
                        help="Optional model revision/tag/commit.")
    parser.add_argument("--warmup_steps", type=int, default=2,
                        help="Warm-up steps prior to timed runs.")

    # Voxtral-specific knobs
    parser.add_argument("--language", type=str, default="en",
                        help="Transcription language hint (e.g. 'en').")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature; 0.0 is recommended for transcription.")

    parser.set_defaults(streaming=True)
    args = parser.parse_args()
    args.streaming = True
    main(args)
