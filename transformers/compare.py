#!/usr/bin/env python3
import os
import json
import sys
import evaluate
from termcolor import colored
import difflib
from tqdm import tqdm

wer_metric = evaluate.load("wer")


def calculate_individual_wer(reference, prediction):
    """Calculate WER for a single example"""
    if not reference.strip():
        return 0.0
    return 100 * wer_metric.compute(references=[reference], predictions=[prediction])


def load_single_file_results(filename, results_dir="results"):
    """Load results from a single file"""
    if not filename.endswith(".jsonl"):
        filename += ".jsonl"

    filepath = os.path.join(results_dir, filename)
    if not os.path.exists(filepath):
        print(colored(f"File not found: {filepath}", "red"))
        return []

    examples = []
    with open(filepath, "r") as f:
        lines = f.readlines()
        for line_num, line in enumerate(tqdm(lines, desc=f"Loading {filename}")):
            if not line.strip():
                continue
            data = json.loads(line)
            data["line_num"] = line_num
            data["wer"] = calculate_individual_wer(data["text"], data["pred_text"])
            examples.append(data)

    return examples


def highlight_differences(reference, prediction):
    """Highlight word-level differences between reference and prediction"""
    ref_words = reference.split()
    pred_words = prediction.split()

    matcher = difflib.SequenceMatcher(None, ref_words, pred_words)

    ref_highlighted = []
    pred_highlighted = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            ref_highlighted.extend(ref_words[i1:i2])
            pred_highlighted.extend(pred_words[j1:j2])
        elif tag == "delete":
            ref_highlighted.extend(
                [colored(word, "red", attrs=["bold"]) for word in ref_words[i1:i2]]
            )
        elif tag == "insert":
            pred_highlighted.extend(
                [colored(word, "green", attrs=["bold"]) for word in pred_words[j1:j2]]
            )
        elif tag == "replace":
            ref_highlighted.extend(
                [colored(word, "red", attrs=["bold"]) for word in ref_words[i1:i2]]
            )
            pred_highlighted.extend(
                [colored(word, "green", attrs=["bold"]) for word in pred_words[j1:j2]]
            )

    return " ".join(ref_highlighted), " ".join(pred_highlighted)


def compare_models(model_a_file, model_b_file, top_n=500):
    """Compare two models and show examples where model A performed worse than model B"""

    print(f"Loading Model A: {colored(model_a_file, 'cyan', attrs=['bold'])}")
    examples_a = load_single_file_results(model_a_file)

    print(f"Loading Model B: {colored(model_b_file, 'magenta', attrs=['bold'])}")
    examples_b = load_single_file_results(model_b_file)

    if not examples_a or not examples_b:
        print(colored("Failed to load one or both model results", "red"))
        return

    if len(examples_a) != len(examples_b):
        print(
            colored(
                f"Warning: Different number of examples (A: {len(examples_a)}, B: {len(examples_b)})",
                "yellow",
            )
        )

    # Find examples where model A performed worse than model B
    worse_examples = []

    min_len = min(len(examples_a), len(examples_b))
    for i in tqdm(range(min_len), desc="Comparing examples"):
        example_a = examples_a[i]
        example_b = examples_b[i]

        # Verify they're the same audio sample
        if example_a.get("text") != example_b.get("text"):
            print(colored(f"Warning: Mismatched reference text at index {i}", "yellow"))
            continue

        if example_a["wer"] > example_b["wer"]:
            comparison_data = {
                "reference": example_a["text"],
                "model_a_pred": example_a["pred_text"],
                "model_b_pred": example_b["pred_text"],
                "model_a_wer": example_a["wer"],
                "model_b_wer": example_b["wer"],
                "wer_diff": example_a["wer"] - example_b["wer"],
                "duration": example_a.get("duration", 0),
                "index": i,
            }
            worse_examples.append(comparison_data)

    if not worse_examples:
        print(
            colored(
                "No examples found where Model A performed worse than Model B", "green"
            )
        )
        return

    # Sort by WER difference (descending - biggest differences first)
    worse_examples.sort(key=lambda x: x["wer_diff"], reverse=True)

    # Take top N
    top_examples = worse_examples[:top_n]

    print(
        f"\nFound {len(worse_examples)} examples where Model A performed worse than Model B"
    )
    print(f"Showing top {len(top_examples)} examples with biggest WER differences\n")
    print("=" * 120)

    for idx, example in enumerate(top_examples, 1):
        ref_highlighted_a, pred_highlighted_a = highlight_differences(
            example["reference"], example["model_a_pred"]
        )
        ref_highlighted_b, pred_highlighted_b = highlight_differences(
            example["reference"], example["model_b_pred"]
        )

        wer_diff_str = f"+{example['wer_diff']:.1f}%"
        model_a_wer_str = f"{example['model_a_wer']:.1f}%"
        model_b_wer_str = f"{example['model_b_wer']:.1f}%"

        print(
            f"\n{colored(f'#{idx}', 'yellow')} | "
            f"Model A WER: {colored(model_a_wer_str, 'red', attrs=['bold'])} | "
            f"Model B WER: {colored(model_b_wer_str, 'green', attrs=['bold'])} | "
            f"Difference: {colored(wer_diff_str, 'red', attrs=['bold'])} | "
            f"Duration: {example['duration']:.1f}s"
        )
        print("-" * 120)

        print(f"{colored('Reference:', 'white', attrs=['bold'])}")
        print(f"  {example['reference']}")

        print(f"\n{colored('Model A Prediction (worse):', 'red', attrs=['bold'])}")
        print(f"  {pred_highlighted_a}")

        print(f"\n{colored('Model B Prediction (better):', 'green', attrs=['bold'])}")
        print(f"  {pred_highlighted_b}")

        print("-" * 120)

        if idx < len(top_examples):
            input(colored("\nPress Enter to see next example...", "grey"))


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare.py <model_a_file> <model_b_file>")
        print("\nExample:")
        print(
            "python compare.py MODEL_jmci-spring-oath-85_DATASET_hf-audio-esb-datasets-test-only-sorted_earnings22_test MODEL_openai-whisper-large-v3_DATASET_hf-audio-esb-datasets-test-only-sorted_earnings22_test"
        )
        sys.exit(1)

    model_a_file = sys.argv[1]
    model_b_file = sys.argv[2]

    compare_models(model_a_file, model_b_file)


if __name__ == "__main__":
    main()
