"""
Experiment 3: Context-dependent rhyming via LLM API.
Tests whether LLMs handle heteronyms (context-dependent pronunciation)
correctly in rhyming tasks, which would require more than static embeddings.
"""

import json
import os
import time
import random
from pathlib import Path
from openai import OpenAI

SEED = 42
random.seed(SEED)

OUTPUT_PATH = Path("results/context_rhyming_results.json")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4.1"

# Heteronyms: words with same spelling but different pronunciation depending on context
HETERONYMS = [
    {
        "word": "read",
        "pron1": {"ipa": "/riːd/", "context": "I will read the book.", "rhymes_with": "seed"},
        "pron2": {"ipa": "/rɛd/", "context": "I have read the book.", "rhymes_with": "said"},
    },
    {
        "word": "lead",
        "pron1": {"ipa": "/liːd/", "context": "She will lead the team.", "rhymes_with": "need"},
        "pron2": {"ipa": "/lɛd/", "context": "The pipe is made of lead.", "rhymes_with": "bed"},
    },
    {
        "word": "wind",
        "pron1": {"ipa": "/wɪnd/", "context": "The wind is blowing hard.", "rhymes_with": "pinned"},
        "pron2": {"ipa": "/waɪnd/", "context": "Wind the clock before bed.", "rhymes_with": "find"},
    },
    {
        "word": "tear",
        "pron1": {"ipa": "/tɪr/", "context": "A tear fell from her eye.", "rhymes_with": "here"},
        "pron2": {"ipa": "/tɛr/", "context": "Don't tear the paper.", "rhymes_with": "bear"},
    },
    {
        "word": "bow",
        "pron1": {"ipa": "/baʊ/", "context": "The actor took a bow.", "rhymes_with": "cow"},
        "pron2": {"ipa": "/boʊ/", "context": "She tied a bow on the gift.", "rhymes_with": "show"},
    },
    {
        "word": "bass",
        "pron1": {"ipa": "/bæs/", "context": "He caught a large bass in the lake.", "rhymes_with": "class"},
        "pron2": {"ipa": "/beɪs/", "context": "The bass guitar sounds deep.", "rhymes_with": "face"},
    },
    {
        "word": "dove",
        "pron1": {"ipa": "/dʌv/", "context": "A white dove flew overhead.", "rhymes_with": "love"},
        "pron2": {"ipa": "/doʊv/", "context": "She dove into the pool.", "rhymes_with": "stove"},
    },
    {
        "word": "minute",
        "pron1": {"ipa": "/ˈmɪnɪt/", "context": "Wait a minute please.", "rhymes_with": "in it"},
        "pron2": {"ipa": "/maɪˈnjuːt/", "context": "The difference is minute.", "rhymes_with": "cute"},
    },
    {
        "word": "close",
        "pron1": {"ipa": "/kloʊz/", "context": "Please close the door.", "rhymes_with": "goes"},
        "pron2": {"ipa": "/kloʊs/", "context": "Stay close to me.", "rhymes_with": "dose"},
    },
    {
        "word": "live",
        "pron1": {"ipa": "/lɪv/", "context": "I live in New York.", "rhymes_with": "give"},
        "pron2": {"ipa": "/laɪv/", "context": "The concert is live tonight.", "rhymes_with": "five"},
    },
    {
        "word": "refuse",
        "pron1": {"ipa": "/rɪˈfjuːz/", "context": "I refuse to accept.", "rhymes_with": "news"},
        "pron2": {"ipa": "/ˈrɛfjuːs/", "context": "Take out the refuse.", "rhymes_with": "goose"},
    },
    {
        "word": "desert",
        "pron1": {"ipa": "/ˈdɛzɝt/", "context": "The Sahara is a vast desert.", "rhymes_with": "dessert"},
        "pron2": {"ipa": "/dɪˈzɝt/", "context": "Don't desert your friends.", "rhymes_with": "alert"},
    },
]


def make_rhyme_query(sentence, word, options):
    """Ask the model which word rhymes with the target in context."""
    prompt = f"""In the following sentence, the word "{word}" is used:
"{sentence}"

Given this specific usage of "{word}", which of the following words rhymes with it?
Options: {', '.join(options)}

Answer with ONLY the rhyming word, nothing else."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=20,
    )
    return response.choices[0].message.content.strip().lower()


def make_simple_rhyme_query(word, options):
    """Ask the model which word rhymes (no context)."""
    prompt = f"""Which of the following words rhymes with "{word}"?
Options: {', '.join(options)}

Answer with ONLY the rhyming word, nothing else."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=20,
    )
    return response.choices[0].message.content.strip().lower()


def run_heteronym_experiment():
    """Test context-dependent rhyming on heteronyms."""
    print(f"Running heteronym experiment with {MODEL}...")
    results = []

    for h in HETERONYMS:
        word = h["word"]
        # Create option set: both correct rhymes + distractors
        correct1 = h["pron1"]["rhymes_with"]
        correct2 = h["pron2"]["rhymes_with"]
        options = [correct1, correct2]

        # Test pronunciation 1 (with context)
        try:
            answer1 = make_rhyme_query(h["pron1"]["context"], word, options)
            correct1_result = correct1.lower() in answer1.lower()
        except Exception as e:
            print(f"  Error for {word} pron1: {e}")
            answer1 = "ERROR"
            correct1_result = False

        # Test pronunciation 2 (with context)
        try:
            answer2 = make_rhyme_query(h["pron2"]["context"], word, options)
            correct2_result = correct2.lower() in answer2.lower()
        except Exception as e:
            print(f"  Error for {word} pron2: {e}")
            answer2 = "ERROR"
            correct2_result = False

        # Test without context (ambiguous)
        try:
            answer_no_ctx = make_simple_rhyme_query(word, options)
        except Exception as e:
            answer_no_ctx = "ERROR"

        result = {
            "word": word,
            "pron1_context": h["pron1"]["context"],
            "pron1_expected": correct1,
            "pron1_answer": answer1,
            "pron1_correct": correct1_result,
            "pron2_context": h["pron2"]["context"],
            "pron2_expected": correct2,
            "pron2_answer": answer2,
            "pron2_correct": correct2_result,
            "no_context_answer": answer_no_ctx,
        }
        results.append(result)

        status1 = "✓" if correct1_result else "✗"
        status2 = "✓" if correct2_result else "✗"
        print(f"  {word}: ctx1={answer1} {status1}, ctx2={answer2} {status2}, no_ctx={answer_no_ctx}")
        time.sleep(0.5)

    return results


def run_standard_rhyme_test():
    """Test standard (unambiguous) rhyming for comparison."""
    print(f"\nRunning standard rhyme test with {MODEL}...")

    test_pairs = [
        ("cat", ["hat", "dog", "run"]),
        ("moon", ["spoon", "sun", "tree"]),
        ("bright", ["night", "day", "cloud"]),
        ("ocean", ["motion", "river", "wave"]),
        ("power", ["flower", "energy", "force"]),
        ("dream", ["stream", "sleep", "night"]),
        ("stone", ["bone", "rock", "hard"]),
        ("rain", ["train", "water", "cloud"]),
        ("light", ["fight", "dark", "lamp"]),
        ("house", ["mouse", "home", "door"]),
        ("blue", ["true", "color", "sky"]),
        ("gold", ["cold", "metal", "yellow"]),
        ("night", ["sight", "dark", "moon"]),
        ("spring", ["ring", "season", "water"]),
        ("heart", ["start", "love", "blood"]),
        # Trickier pairs (orthographic non-rhymes that phonetically rhyme)
        ("tough", ["stuff", "dough", "cough"]),
        ("weight", ["late", "height", "freight"]),
        ("through", ["blue", "rough", "cough"]),
        ("said", ["bed", "paid", "maid"]),
        ("blood", ["mud", "good", "food"]),
    ]

    results = []
    for word, options in test_pairs:
        try:
            answer = make_simple_rhyme_query(word, options)
            expected = options[0]  # First option is always the rhyme
            correct = expected.lower() in answer.lower()
        except Exception as e:
            answer = "ERROR"
            correct = False

        results.append({
            "word": word,
            "options": options,
            "expected": options[0],
            "answer": answer,
            "correct": correct,
        })
        status = "✓" if correct else "✗"
        print(f"  {word} → {answer} (expected: {options[0]}) {status}")
        time.sleep(0.3)

    return results


def main():
    heteronym_results = run_heteronym_experiment()
    standard_results = run_standard_rhyme_test()

    # Compute summary statistics
    het_ctx_correct = sum(1 for r in heteronym_results if r["pron1_correct"]) + \
                      sum(1 for r in heteronym_results if r["pron2_correct"])
    het_ctx_total = len(heteronym_results) * 2

    std_correct = sum(1 for r in standard_results if r["correct"])
    std_total = len(standard_results)

    # Split standard into easy (first 15) and tricky (last 5)
    easy_correct = sum(1 for r in standard_results[:15] if r["correct"])
    tricky_correct = sum(1 for r in standard_results[15:] if r["correct"])

    summary = {
        "model": MODEL,
        "heteronym_accuracy": het_ctx_correct / het_ctx_total,
        "heteronym_correct": het_ctx_correct,
        "heteronym_total": het_ctx_total,
        "standard_accuracy": std_correct / std_total,
        "standard_correct": std_correct,
        "standard_total": std_total,
        "easy_rhyme_accuracy": easy_correct / 15,
        "tricky_rhyme_accuracy": tricky_correct / 5,
    }

    output = {
        "summary": summary,
        "heteronym_results": heteronym_results,
        "standard_results": standard_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Heteronym (context-dependent): {het_ctx_correct}/{het_ctx_total} = {summary['heteronym_accuracy']:.1%}")
    print(f"Standard rhyming: {std_correct}/{std_total} = {summary['standard_accuracy']:.1%}")
    print(f"  Easy pairs: {easy_correct}/15 = {summary['easy_rhyme_accuracy']:.1%}")
    print(f"  Tricky pairs: {tricky_correct}/5 = {summary['tricky_rhyme_accuracy']:.1%}")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
