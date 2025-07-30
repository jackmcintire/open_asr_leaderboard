#!/usr/bin/env python3
import os
import json
import sys
import argparse
from collections import defaultdict
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


def load_model_results(model_identifier, results_dir="results"):
    """Load all results for a given model or a single file."""
    all_examples = []
    relevant_files = []

    # Heuristic: if it contains _DATASET_, user might be specifying a file.
    is_filename_like = "_DATASET_" in model_identifier

    if is_filename_like:
        filename_with_ext = model_identifier
        if not filename_with_ext.endswith(".jsonl"):
            filename_with_ext += ".jsonl"

        filepath = os.path.join(results_dir, filename_with_ext)
        if os.path.exists(filepath):
            relevant_files.append(filename_with_ext)
            print(f"Loading from specific file: {filename_with_ext}")

    # If it wasn't a file-like string or the file wasn't found,
    # treat it as a model name.
    if not relevant_files:
        model_name = model_identifier
        for filename in os.listdir(results_dir):
            if filename.startswith(
                f"MODEL_{model_name}_DATASET_"
            ) and filename.endswith(".jsonl"):
                relevant_files.append(filename)

        if relevant_files:
            print(f"Found {len(relevant_files)} dataset files for model '{model_name}'")

    # Process each file with progress bar
    for filename in tqdm(relevant_files, desc="Loading datasets"):
        # Extract dataset name from filename
        try:
            dataset_name = filename.split("_DATASET_")[1].replace(".jsonl", "")
        except IndexError:
            dataset_name = "unknown"

        filepath = os.path.join(results_dir, filename)
        with open(filepath, "r") as f:
            lines = f.readlines()
            for line_num, line in enumerate(
                tqdm(lines, desc=f"Processing {dataset_name}", leave=False)
            ):
                if not line.strip():
                    continue
                data = json.loads(line)
                data["dataset"] = dataset_name
                data["line_num"] = line_num
                data["wer"] = calculate_individual_wer(data["text"], data["pred_text"])
                all_examples.append(data)

    return all_examples


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


def visualize_worst_examples(model_identifier, top_n=50, interactive=True):
    """Visualize the worst WER examples for a model or a specific file"""
    print(f"\nLoading results for: {colored(model_identifier, 'cyan', attrs=['bold'])}")

    examples = load_model_results(model_identifier)

    if not examples:
        print(colored(f"No results found for '{model_identifier}'", "red"))
        return

    print(f"\nSorting {len(examples)} examples by WER...")
    # Sort by WER (descending)
    examples.sort(key=lambda x: x["wer"], reverse=True)

    # Take top N
    worst_examples = examples[:top_n]

    print(f"\nFound {len(examples)} total examples across all datasets")
    print(f"Showing top {top_n} examples with highest WER\n")
    print("=" * 120)

    for idx, example in enumerate(worst_examples, 1):
        ref_highlighted, pred_highlighted = highlight_differences(
            example["text"], example["pred_text"]
        )

        wer_str = f"{example['wer']:.1f}%"
        print(
            f"\n{colored(f'#{idx}', 'yellow')} | Dataset: {colored(example['dataset'], 'blue')} | "
            f"WER: {colored(wer_str, 'red', attrs=['bold'])} | "
            f"Duration: {example['duration']:.1f}s"
        )
        print("-" * 120)

        print(f"{colored('Reference:', 'white', attrs=['bold'])}")
        print(f"  {ref_highlighted}")

        print(f"\n{colored('Prediction:', 'white', attrs=['bold'])}")
        print(f"  {pred_highlighted}")

        print("-" * 120)

        if interactive and idx < len(worst_examples):
            input(colored("\nPress Enter to see next example...", "grey"))
        elif not interactive and idx < len(worst_examples):
            print()  # Just add a blank line between examples


def main():
    parser = argparse.ArgumentParser(
        description="Visualize worst WER examples for a model"
    )
    parser.add_argument(
        "model_or_filename", nargs="?", help="Name of the model or filename to analyze"
    )
    parser.add_argument(
        "-n",
        "--top",
        type=int,
        default=50,
        help="Number of worst examples to show (default: 50)",
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List all available models"
    )
    parser.add_argument(
        "--list-files", action="store_true", help="List all available result files"
    )
    parser.add_argument(
        "--no-interactive", action="store_true", help="Don't pause between examples"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing the results files (default: results)",
    )

    args = parser.parse_args()

    if args.list_models:
        # List all available models
        models = set()
        for filename in os.listdir(args.results_dir):
            if filename.startswith("MODEL_") and filename.endswith(".jsonl"):
                model_name = filename[6:].split("_DATASET_")[0]
                models.add(model_name)

        print("\nAvailable models:")
        for model in sorted(models):
            print(f"  - {model}")
        return

    if args.list_files:
        # List all available result files
        print("\nAvailable result files:")
        for filename in sorted(os.listdir("results")):
            if filename.startswith("MODEL_") and filename.endswith(".jsonl"):
                print(f"  - {filename}")
        return

    if not args.model_or_filename:
        parser.error(
            "model_or_filename is required unless using --list-models or --list-files"
        )

    visualize_worst_examples(
        args.model_or_filename,
        top_n=args.top,
        interactive=not args.no_interactive,
    )


if __name__ == "__main__":
    main()
