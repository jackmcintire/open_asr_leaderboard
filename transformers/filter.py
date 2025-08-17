import sys, pathlib

ROOT = pathlib.Path().resolve().parent
sys.path.append(str(ROOT))

import json
import os

from normalizer.data_utils import normalizer as data_utils_normalizer

normalizer = data_utils_normalizer

# Cell 1
from datasets import load_dataset, Dataset, DatasetDict, Value
from pathlib import Path
import csv, re

from datasets import load_dataset, DatasetDict, Value
from pathlib import Path
import csv, re

SPECIAL_WORDS_PATH = (
    "/home/ubuntu/jtm/open_asr_leaderboard/transformers/global_dict_500.csv"
)

with open(SPECIAL_WORDS_PATH, "r", encoding="utf-8") as f:
    special_words = [line.strip().rstrip(",") for line in f if line.strip()]
    normalized_special_words = {normalizer(word) for word in special_words}


def has_special_word(x):
    text = normalizer(x["text"])
    words_in_text = set(text.split())
    matched_words = words_in_text.intersection(normalized_special_words)
    return len(matched_words) > 0


ds_stream = load_dataset(
    "aquavoice/cleaned_dataset_full_2x_en_resplit",
    split="test",
    streaming=True,  # <- important
)

print("Filtering dataset...")
filtered_stream = ds_stream.filter(has_special_word)

features = ds_stream.features
filtered_test = Dataset.from_generator(
    lambda: (ex for ex in filtered_stream), features=features
)


new_ds_name = "jmci/aquavoice_cleaned_dataset_full_2x_en_resplit_filtered"
to_push = DatasetDict({"test": filtered_test})
to_push.push_to_hub(new_ds_name, private=True, max_shard_size="1GB")
