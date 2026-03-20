"""
Data preparation: Build word→IPA mappings for single-token words.
Uses Pythia-1.4B tokenizer (ungated, TransformerLens supported).
Outputs JSON mapping token_id → {word, ipa_phonemes, ipa_string}.
"""

import json
from pathlib import Path
from collections import defaultdict

WIKIPRON_PATH = Path("datasets/wikipron/data/scrape/tsv/eng_latn_us_broad.tsv")
OXFORD_PATH = Path("datasets/oxford_5000/full-word.json")
OUTPUT_PATH = Path("results/token_phoneme_data.json")

MODEL_NAME = "gpt2-medium"


def load_wikipron(path):
    """Load WikiPron TSV: word → list of IPA transcriptions."""
    word2ipa = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                word = parts[0].lower().strip("'")
                ipa = parts[1].strip()
                word2ipa[word].append(ipa)
    return word2ipa


def load_oxford_words(path):
    """Load Oxford 5000 word list."""
    with open(path, "r") as f:
        data = json.load(f)
    return {entry["value"]["word"].lower() for entry in data}


def ipa_to_phoneme_list(ipa_str):
    """Convert space-separated IPA to list of phoneme symbols."""
    return [p.strip() for p in ipa_str.split() if p.strip()]


def get_single_token_words(tokenizer):
    """Find words that are encoded as a single token."""
    vocab = tokenizer.get_vocab()
    single_token_words = {}

    for token_str, token_id in vocab.items():
        # GPT-NeoX/Pythia uses Ġ for space prefix
        clean = token_str.replace("Ġ", "").replace("▁", "").strip()
        if not clean or not clean.isalpha() or len(clean) < 2:
            continue

        # Check if word encodes to single token (with space prefix)
        encoded = tokenizer.encode(" " + clean.lower(), add_special_tokens=False)
        if len(encoded) == 1:
            single_token_words[clean.lower()] = encoded[0]

    return single_token_words


def build_dataset():
    """Build the word → phoneme dataset for single-token words."""
    from transformers import AutoTokenizer

    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading WikiPron pronunciations...")
    word2ipa = load_wikipron(WIKIPRON_PATH)
    print(f"  WikiPron: {len(word2ipa)} words")

    print("Loading Oxford 5000...")
    oxford_words = load_oxford_words(OXFORD_PATH)
    print(f"  Oxford: {len(oxford_words)} words")

    print("Finding single-token words...")
    single_token_words = get_single_token_words(tokenizer)
    print(f"  Single-token words: {len(single_token_words)}")

    # Build dataset: single-token words with IPA transcriptions
    dataset = []
    all_phonemes = set()

    for word, token_id in single_token_words.items():
        if word in word2ipa:
            ipa_str = word2ipa[word][0]  # Use first transcription
            phonemes = ipa_to_phoneme_list(ipa_str)
            if phonemes:
                all_phonemes.update(phonemes)
                dataset.append({
                    "word": word,
                    "token_id": token_id,
                    "ipa_string": ipa_str,
                    "ipa_phonemes": phonemes,
                    "in_oxford": word in oxford_words
                })

    phoneme_list = sorted(all_phonemes)
    phoneme_to_idx = {p: i for i, p in enumerate(phoneme_list)}

    # Add multi-hot encoding
    for item in dataset:
        multi_hot = [0] * len(phoneme_list)
        for p in item["ipa_phonemes"]:
            multi_hot[phoneme_to_idx[p]] = 1
        item["multi_hot"] = multi_hot

    result = {
        "model_name": MODEL_NAME,
        "phoneme_list": phoneme_list,
        "phoneme_to_idx": phoneme_to_idx,
        "num_phonemes": len(phoneme_list),
        "num_words": len(dataset),
        "words": dataset
    }

    print(f"\nDataset built:")
    print(f"  Words with IPA: {len(dataset)}")
    print(f"  Unique phonemes: {len(phoneme_list)}")
    print(f"  Oxford words: {sum(1 for d in dataset if d['in_oxford'])}")
    print(f"  Sample phonemes: {phoneme_list[:20]}")
    print(f"  Sample words: {[d['word'] for d in dataset[:10]]}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {OUTPUT_PATH}")

    return result


if __name__ == "__main__":
    build_dataset()
