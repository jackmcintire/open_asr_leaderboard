# run_voxtral.py  (streaming-friendly, batched, assumes 16 kHz)
import argparse
import os
import time
from typing import Iterable, List, Optional

import torch
from transformers import AutoProcessor

try:
    from transformers import VoxtralForConditionalGeneration
except Exception:
    raise RuntimeError(
        "Please update `transformers` to a version that includes VoxtralForConditionalGeneration."
    )

import evaluate
from normalizer import data_utils
from tqdm import tqdm

wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision("high")


def _device_from_int(index: int) -> torch.device:
    if index is not None and index >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def _iter_streaming_batches(
    ds,
    batch_size: int,
    max_eval_samples: Optional[int],
) -> Iterable[List[dict]]:
    """
    Iterate over a (possibly streaming) dataset and yield lists of examples,
    without relying on __len__ or random access.

    Any max_eval_samples <= 0 is treated as "unbounded".
    """
    limit = None if (max_eval_samples is None or max_eval_samples <= 0) else int(max_eval_samples)
    batch: List[dict] = []
    seen = 0

    for ex in ds:
        if limit is not None and seen >= limit:
            break
        batch.append(ex)
        seen += 1
        if len(batch) == batch_size:
            yield batch
            batch = []

    # Flush tail
    if batch:
        yield batch


def _prompt_lengths_per_sample(inputs, processor) -> List[int]:
    """
    Compute per-sample prompt lengths for trimming before decoding.
    Prefer attention_mask; fallback to non-pad counts; otherwise max length.
    """
    input_ids = inputs["input_ids"]
    B, T = input_ids.shape

    if "attention_mask" in inputs and inputs["attention_mask"] is not None:
        mask = inputs["attention_mask"]
        lens = mask.sum(dim=1).tolist()
        return [int(x) for x in lens]

    pad_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)
    if pad_id is not None:
        lens = (input_ids != pad_id).sum(dim=1).tolist()
        return [int(x) for x in lens]

    return [T] * B


def main(args):
    # Normalize "all samples" signal
    if args.max_eval_samples is not None and args.max_eval_samples <= 0:
        args.max_eval_samples = None

    device = _device_from_int(args.device)
    print(f"Using device: {device}")

    # Results manifest path (ESB-friendly)
    model_id_str = args.model_id.replace("/", "-")
    dataset_path_str = args.dataset_path.replace("/", "-")
    dataset_str = args.dataset.replace("/", "-")
    manifest_path = (
        f"results/MODEL_{model_id_str}_DATASET_{dataset_path_str}_"
        f"{dataset_str}_{args.split}.jsonl"
    )
    if os.path.exists(manifest_path) and not args.overwrite:
        print(f"Results already exist at {manifest_path} — skipping. Use --overwrite to re-run.")
        return

    # Voxtral + processor
    processor = AutoProcessor.from_pretrained(args.model_id, revision=args.revision)
    device_map = "cuda" if device.type == "cuda" else "cpu"
    model = VoxtralForConditionalGeneration.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    gen_kwargs = {"max_new_tokens": args.max_new_tokens or 500}
    lang = args.language
    assumed_sr = 16_000  # per your assumption

    # Load dataset (streaming honored via args.streaming used by data_utils)
    dataset = data_utils.load_data(args)
    dataset = data_utils.prepare_data(dataset)

    # Accumulators
    all_refs: List[str] = []
    all_preds: List[str] = []
    all_audio_lengths_s: List[float] = []
    all_transcription_times_s: List[float] = []
    total_audio_sec = 0.0
    total_time_sec = 0.0

    # Iterate over streaming batches
    batches = _iter_streaming_batches(dataset, args.batch_size, args.max_eval_samples)
    for batch in tqdm(batches, desc="Batches"):
        audio_arrays, secs, refs = [], [], []

        for ex in batch:
            wave = ex["audio"]["array"]
            audio_arrays.append(wave)
            secs.append(len(wave) / float(assumed_sr))
            refs.append(ex["norm_text"])

        # Single batched transcription request
        req = dict(
            audio=audio_arrays,
            sampling_rate=assumed_sr,
            format=[args.voxtral_format] * len(batch),  # one container per sample
            model_id=args.model_id,
            return_tensors="pt",
        )
        if lang is not None:
            req["language"] = lang

        # Time preprocessing + generate + decode
        start = time.time()
        inputs = processor.apply_transcription_request(**req)
        inputs = inputs.to(device, dtype=torch.bfloat16)  # only float tensors are cast

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        # Per-sample prompt trimming
        prompt_lens = _prompt_lengths_per_sample(inputs, processor)
        trimmed_ids = [outputs[i, prompt_lens[i]:].tolist() for i in range(len(batch))]

        decoded = processor.batch_decode(trimmed_ids, skip_special_tokens=True)
        elapsed = time.time() - start

        # Metrics
        per_sample_time = elapsed / len(batch)
        all_transcription_times_s.extend([per_sample_time] * len(batch))
        all_audio_lengths_s.extend(secs)
        total_time_sec += elapsed
        total_audio_sec += sum(secs)

        all_preds.extend([data_utils.normalizer(p) for p in decoded])
        all_refs.extend(refs)

    # Write manifest + final metrics
    manifest_path = data_utils.write_manifest(
        all_refs,
        all_preds,
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=all_audio_lengths_s,
        transcription_time=all_transcription_times_s,
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    wer = wer_metric.compute(references=all_refs, predictions=all_preds)
    wer = round(100 * wer, 2)
    rtfx = round(total_audio_sec / total_time_sec, 2) if total_time_sec > 0 else 0.0
    print("WER:", wer, "%", "RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="esb/datasets")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)

    # Streaming toggle (ON by default here)
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Disable HF datasets streaming mode.",
    )
    parser.set_defaults(streaming=True)

    # Voxtral-specific
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--voxtral_format", type=str, default="WAV")

    # Re-run even if results file exists
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    main(args)
