import os
import json
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
    dataset_name = (
        dataset_parts[1]
        if dataset_parts[0] == "hf-audio-esb-datasets-test-only-sorted"
        else dataset_parts[0]
    )

    return model_name, dataset_name


def compute_metrics(filepath):
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

    wer = 100 * wer_metric.compute(references=references, predictions=predictions)
    rtfx = sum(audio_durations) / sum(transcription_times)

    return {
        "wer": round(wer, 2),
        "rtfx": round(rtfx, 2),
        "num_samples": len(data),
        "total_audio_hours": round(sum(audio_durations) / 3600, 2),
        "special_words_acc": round(special_words_accuracy, 2),
        "samples_with_special": total_with_special,
    }


def main():
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
            os.path.join(results_dir, filename)
        )

    print("\n" + "=" * 80)
    print("ASR Evaluation Results")
    print("=" * 80)

    for model_name, datasets in model_results.items():
        print(f"\n{model_name}")
        print("-" * 60)

        dataset_count = 0  # number of datasets seen
        sum_wer = 0.0  # un-weighted sum of dataset WERs
        total_samples = 0  # running sample count
        weighted_wer = 0.0  # running Σ(wer * samples)

        for dataset, m in sorted(datasets.items()):
            print(
                f"  {dataset:20s} | WER: {m['wer']:5.1f}% | RTFx: {m['rtfx']:6.1f}x | "
                f"Special: {m['special_words_acc']:5.1f}% | "
                f"Samples: {m['num_samples']:5d} | Hours: {m['total_audio_hours']:5.1f}h"
            )
            if (
                not "global-dict" in dataset
            ):  # exclude from metrics as these are all present in the aquavoice dataset
                dataset_count += 1
                sum_wer += m["wer"]
                total_samples += m["num_samples"]
                weighted_wer += m["wer"] * m["num_samples"]

        if dataset_count:
            avg_wer = sum_wer / dataset_count
            weighted_avg = (weighted_wer / total_samples) if total_samples else avg_wer
            print(f"\n  {'OVERALL':20s} | WER: {avg_wer:5.1f}% ({weighted_avg:5.1f}%)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
