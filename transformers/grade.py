import os
import json
import argparse
from collections import defaultdict
import evaluate

wer_metric = evaluate.load("wer")


def parse_filename(filename):
    if not (filename.startswith("MODEL_") and filename.endswith(".jsonl")):
        return None, None

    core = filename[6:-6]
    parts = core.split("_DATASET_")
    if len(parts) != 2:
        return None, None

    model_name = parts[0]
    dataset_parts = parts[1].split("_")
    # For HF Audio ESB datasets, the structure is:
    #   hf-audio-esb-datasets-test-only-sorted_<dataset>_<subset>
    # e.g., hf-audio-esb-datasets-test-only-sorted_librispeech_test.clean
    if dataset_parts[0] == "hf-audio-esb-datasets-test-only-sorted":
        base_dataset = dataset_parts[1] if len(dataset_parts) > 1 else ""
        # Special-case Librispeech to distinguish clean vs other without showing "test"
        # Filenames carry subset like "test.clean" or "test.other"
        if base_dataset == "librispeech" and len(dataset_parts) > 2:
            subset = dataset_parts[2]  # e.g., "test.clean"
            # Extract variant after the dot if present
            variant = subset.split(".")[1] if "." in subset else subset
            dataset_name = f"librispeech_{variant}"
        else:
            # Keep just the dataset name (e.g., "earnings22") as before
            dataset_name = base_dataset
    else:
        dataset_name = dataset_parts[0]

    return model_name, dataset_name


def compute_metrics(filepath, duration_threshold=None):
    data = [json.loads(line) for line in open(filepath) if line.strip()]

    references = [d["text"] for d in data]
    predictions = [d["pred_text"] for d in data]
    audio_durations = [d["duration"] for d in data]
    transcription_times = [d["time"] for d in data]
    special_words = [
        d.get("matched_special_words", []) for d in data
    ]  # Use get() with default empty list

    # Calculate special words accuracy
    special_words_correct = 0
    total_with_special = 0
    for pred, special in zip(predictions, special_words):
        if special:  # Only count examples that have special words
            total_with_special += 1
            # Split prediction into words and convert to lowercase
            pred_words = set(pred.lower().split())
            # Check if all special words are in prediction as whole words
            if all(word.lower() in pred_words for word in special):
                special_words_correct += 1

    special_words_accuracy = (
        (special_words_correct / total_with_special * 100)
        if total_with_special > 0
        else 0
    )

    audio_seconds = sum(audio_durations)
    transcription_seconds = sum(transcription_times)
    wer = 100 * wer_metric.compute(references=references, predictions=predictions)
    rtfx = audio_seconds / transcription_seconds if transcription_seconds else 0.0

    result = {
        "wer": round(wer, 2),
        "rtfx": round(rtfx, 2),
        "num_samples": len(data),
        "total_audio_hours": round(audio_seconds / 3600, 2),
        "special_words_acc": round(special_words_accuracy, 2),
        "samples_with_special": total_with_special,
        "audio_seconds": audio_seconds,
        "transcription_seconds": transcription_seconds,
    }
    
    # Calculate metrics for clips longer than threshold if specified
    if duration_threshold is not None:
        long_refs = []
        long_preds = []
        for i, duration in enumerate(audio_durations):
            if duration > duration_threshold:
                long_refs.append(references[i])
                long_preds.append(predictions[i])
        
        if long_refs:
            long_wer = 100 * wer_metric.compute(references=long_refs, predictions=long_preds)
            result["long_clip_wer"] = round(long_wer, 2)
            result["long_clip_samples"] = len(long_refs)
        else:
            result["long_clip_wer"] = None
            result["long_clip_samples"] = 0
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Grade ASR model results")
    parser.add_argument(
        "--min-duration",
        type=float,
        help="Minimum duration in seconds to calculate separate WER for long clips"
    )
    args = parser.parse_args()
    
    results_dir = os.path.abspath("results")

    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    model_results = defaultdict(dict)

    for filename in sorted(os.listdir(results_dir)):
        print(filename)
        if not filename.endswith(".jsonl"):
            continue

        model_name, dataset_name = parse_filename(filename)
        if not model_name:
            print(f"Warning: Could not parse {filename}")
            continue

        print(f"Processing {model_name} on {dataset_name}...")
        model_results[model_name][dataset_name] = compute_metrics(
            os.path.join(results_dir, filename),
            duration_threshold=args.min_duration
        )

    print("\n" + "=" * 80)
    print("ASR Evaluation Results")
    if args.min_duration:
        print(f"(Including WER for clips > {args.min_duration} seconds)")
    print("=" * 80)

    # If we need long clip metrics, collect aggregated data
    model_long_clip_data = defaultdict(lambda: {"refs": [], "preds": []})
    
    if args.min_duration:
        # Re-read all files to collect long clip data for aggregation
        for filename in sorted(os.listdir(results_dir)):
            if not filename.endswith(".jsonl"):
                continue
            
            model_name, dataset_name = parse_filename(filename)
            if not model_name:
                continue
                
            # Skip excluded datasets
            if "global-dict" in dataset_name or "aquavoice" in dataset_name:
                continue
                
            # Read the file and collect long clips
            filepath = os.path.join(results_dir, filename)
            with open(filepath) as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        if d["duration"] > args.min_duration:
                            model_long_clip_data[model_name]["refs"].append(d["text"])
                            model_long_clip_data[model_name]["preds"].append(d["pred_text"])

    for model_name, datasets in model_results.items():
        print(f"\n{model_name}")
        print("-" * 60)

        dataset_count = 0  # number of datasets seen
        sum_wer = 0.0  # un-weighted sum of dataset WERs
        total_samples = 0  # running sample count
        weighted_wer = 0.0  # running Σ(wer * samples)
        total_audio_sec = 0.0
        total_trans_sec = 0.0

        for dataset, m in sorted(datasets.items()):
            # Build the main line
            line = (
                f"  {dataset:20s} | WER: {m['wer']:5.2f}% | RTFx: {m['rtfx']:6.1f}x | "
                f"Special: {m['special_words_acc']:5.1f}% | "
                f"Samples: {m['num_samples']:5d} | Hours: {m['total_audio_hours']:5.1f}h"
            )
            
            # Add long clip WER if available
            if args.min_duration and "long_clip_wer" in m and m["long_clip_wer"] is not None:
                line += f" | Long WER: {m['long_clip_wer']:5.1f}% ({m['long_clip_samples']} samples)"
            
            print(line)
            
            if (
                not "global-dict" in dataset and not "aquavoice" in dataset
            ):  # exclude from metrics as these are all present in the aquavoice dataset
                dataset_count += 1
                sum_wer += m["wer"]
                total_samples += m["num_samples"]
                weighted_wer += m["wer"] * m["num_samples"]
                total_audio_sec += m.get("audio_seconds", 0.0)
                total_trans_sec += m.get("transcription_seconds", 0.0)

        if dataset_count:
            avg_wer = sum_wer / dataset_count
            weighted_avg = (weighted_wer / total_samples) if total_samples else avg_wer
            overall_rtfx = (
                (total_audio_sec / total_trans_sec) if total_trans_sec else 0.0
            )
            
            # Build overall line
            overall_line = f"\n  {'OVERALL':20s} | WER: {avg_wer:5.2f}% ({weighted_avg:5.1f}%) | RTFx: {overall_rtfx:6.1f}x"
            
            # Add total long clip WER if available
            if args.min_duration and model_name in model_long_clip_data:
                long_data = model_long_clip_data[model_name]
                if long_data["refs"]:
                    total_long_wer = 100 * wer_metric.compute(
                        references=long_data["refs"],
                        predictions=long_data["preds"]
                    )
                    overall_line += f" | Total Long WER: {total_long_wer:5.2f}% ({len(long_data['refs'])} samples)"
            
            print(overall_line)

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
