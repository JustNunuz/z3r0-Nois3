"""
z3ro nois3: Tokenizer Forensic Benchmark (V7)
----------------------------------------------
Head-to-head comparison of BPE, WordPiece, SentencePiece, and Bridge
across English and Shona text with cost analysis.

Usage:
    python tokenizer_bench.py [shona_file] [english_file]
"""

import tiktoken
import sentencepiece as spm
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import os
import sys
import tempfile

from engine import BantuBridgeEngine


# ── Cost constants (GPT-4 input pricing) ─────────────────────────
COST_PER_1K_TOKENS = 0.03  # $0.03 per 1K tokens (GPT-4 input)


def count_words(text: str) -> int:
    return len(text.split())


def bpe_tokenize(text: str) -> int:
    """Tokenize with OpenAI's cl100k_base (GPT-4 BPE)."""
    enc = tiktoken.encoding_for_model("gpt-4")
    return len(enc.encode(text))


def wordpiece_tokenize(text: str) -> int:
    """
    Train a WordPiece tokenizer on the text itself, then tokenize.
    This simulates what BERT would do with a Shona-aware vocabulary.
    """
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.WordPieceTrainer(
        vocab_size=30000,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    )

    # Train on the text
    lines = text.split("\n")
    tokenizer.train_from_iterator(lines, trainer=trainer)

    encoded = tokenizer.encode(text)
    return len(encoded.ids)


def sentencepiece_tokenize(text: str) -> int:
    """
    Train a SentencePiece (Unigram) model on the text, then tokenize.
    SentencePiece is language-agnostic and often better for non-Latin.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(text)
        tmp_path = tmp.name

    model_prefix = tmp_path + "_sp"

    try:
        # Auto-adapt vocab size to corpus size to avoid crash on small texts
        unique_chars = len(set(text))
        max_vocab = min(16000, max(100, unique_chars - 10))

        spm.SentencePieceTrainer.Train(
            input=tmp_path,
            model_prefix=model_prefix,
            vocab_size=max_vocab,
            model_type="unigram",
            character_coverage=0.9995,
            num_threads=1,
            train_extremely_large_corpus=False,
        )

        sp = spm.SentencePieceProcessor()
        sp.Load(model_prefix + ".model")
        pieces = sp.Encode(text, out_type=str)
        token_count = len(pieces)
    except RuntimeError as e:
        # SentencePiece crashes if the corpus is extremely small (e.g., a single line).
        # In this case, we fallback to WordPiece which handles small texts better.
        print(f"    [WARN] SentencePiece training failed (corpus too small?). Falling back to WordPiece. Error: {e}")
        token_count = wordpiece_tokenize(text)
    finally:
        # Cleanup temp files
        for ext in ["", "_sp.model", "_sp.vocab"]:
            path = tmp_path + ext if ext else tmp_path
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    return token_count


def bridge_tokenize(text: str, corpus_text: str) -> int:
    """
    Apply the Bridge adaptive compression, then tokenize with BPE.
    This measures the Bridge's real-world impact on GPT-4 costs.
    """
    engine = BantuBridgeEngine()
    engine.learn(corpus_text, top_n=150)
    optimized = engine.process_text(text)

    enc = tiktoken.encoding_for_model("gpt-4")
    return len(enc.encode(optimized))


def run_benchmark(shona_text: str, english_text: str):
    """Run the full forensic comparison and print results."""
    shona_words = count_words(shona_text)
    english_words = count_words(english_text)

    print("=" * 78)
    print("  z3ro nois3: Tokenizer Forensic Benchmark (V7)")
    print("=" * 78)
    print(f"\n  Shona corpus  : {shona_words:,} words")
    print(f"  English corpus: {english_words:,} words")
    print()

    # ── English tokenization ─────────────────────────────────────
    print("  [1/7] BPE (English)...")
    en_bpe = bpe_tokenize(english_text)

    print("  [2/7] WordPiece (English)...")
    en_wp = wordpiece_tokenize(english_text)

    print("  [3/7] SentencePiece (English)...")
    en_sp = sentencepiece_tokenize(english_text)

    # ── Shona tokenization ───────────────────────────────────────
    print("  [4/7] BPE (Shona)...")
    sn_bpe = bpe_tokenize(shona_text)

    print("  [5/7] WordPiece (Shona)...")
    sn_wp = wordpiece_tokenize(shona_text)

    print("  [6/7] SentencePiece (Shona)...")
    sn_sp = sentencepiece_tokenize(shona_text)

    print("  [7/7] Bridge + BPE (Shona)...")
    sn_bridge = bridge_tokenize(shona_text, shona_text)

    # ── Compute metrics ──────────────────────────────────────────
    def fertility(tokens, words):
        return tokens / words if words else 0

    def cost(tokens):
        return tokens * COST_PER_1K_TOKENS / 1000

    def tax_vs_english(sn_fert, en_fert):
        return sn_fert / en_fert if en_fert else 0

    en_bpe_f = fertility(en_bpe, english_words)

    results = [
        {
            "Tokenizer": "BPE (GPT-4)",
            "Lang": "English",
            "Tokens": en_bpe,
            "Fertility": fertility(en_bpe, english_words),
            "Cost": cost(en_bpe),
            "Tax": 1.0,
        },
        {
            "Tokenizer": "WordPiece",
            "Lang": "English",
            "Tokens": en_wp,
            "Fertility": fertility(en_wp, english_words),
            "Cost": cost(en_wp),
            "Tax": tax_vs_english(fertility(en_wp, english_words), en_bpe_f),
        },
        {
            "Tokenizer": "SentencePiece",
            "Lang": "English",
            "Tokens": en_sp,
            "Fertility": fertility(en_sp, english_words),
            "Cost": cost(en_sp),
            "Tax": tax_vs_english(fertility(en_sp, english_words), en_bpe_f),
        },
        {
            "Tokenizer": "BPE (GPT-4)",
            "Lang": "Shona",
            "Tokens": sn_bpe,
            "Fertility": fertility(sn_bpe, shona_words),
            "Cost": cost(sn_bpe),
            "Tax": tax_vs_english(fertility(sn_bpe, shona_words), en_bpe_f),
        },
        {
            "Tokenizer": "WordPiece",
            "Lang": "Shona",
            "Tokens": sn_wp,
            "Fertility": fertility(sn_wp, shona_words),
            "Cost": cost(sn_wp),
            "Tax": tax_vs_english(fertility(sn_wp, shona_words), en_bpe_f),
        },
        {
            "Tokenizer": "SentencePiece",
            "Lang": "Shona",
            "Tokens": sn_sp,
            "Fertility": fertility(sn_sp, shona_words),
            "Cost": cost(sn_sp),
            "Tax": tax_vs_english(fertility(sn_sp, shona_words), en_bpe_f),
        },
        {
            "Tokenizer": "Bridge + BPE",
            "Lang": "Shona",
            "Tokens": sn_bridge,
            "Fertility": fertility(sn_bridge, shona_words),
            "Cost": cost(sn_bridge),
            "Tax": tax_vs_english(fertility(sn_bridge, shona_words), en_bpe_f),
        },
    ]

    # ── Print results ────────────────────────────────────────────
    print(f"\n{'─' * 78}")
    print(f"  {'Tokenizer':<20} {'Lang':<8} {'Tokens':>10} {'Fertility':>10} {'Cost ($)':>10} {'Tax vs EN':>10}")
    print(f"{'─' * 78}")
    for r in results:
        print(
            f"  {r['Tokenizer']:<20} {r['Lang']:<8} {r['Tokens']:>10,} {r['Fertility']:>10.2f} {r['Cost']:>10.4f} {r['Tax']:>9.2f}x"
        )
    print(f"{'─' * 78}")

    # ── Bridge savings summary ───────────────────────────────────
    bridge_savings_tokens = sn_bpe - sn_bridge
    bridge_savings_pct = (bridge_savings_tokens / sn_bpe * 100) if sn_bpe else 0
    bridge_savings_cost = cost(sn_bpe) - cost(sn_bridge)

    print(f"\n  BRIDGE IMPACT (vs raw BPE on Shona)")
    print(f"{'─' * 78}")
    print(f"  Tokens saved       : {bridge_savings_tokens:,}")
    print(f"  Savings percentage : {bridge_savings_pct:.2f}%")
    print(f"  Cost saved (per run): ${bridge_savings_cost:.4f}")
    print(f"  Cost saved (per 1M tokens): ${bridge_savings_cost * 1000000 / sn_bpe:.2f}")
    print(f"{'=' * 78}\n")

    return results


# ── Default English reference text ───────────────────────────────
DEFAULT_ENGLISH = """
The people of Zimbabwe are hardworking and resilient. They wake up every morning
and go to work despite the challenges they face. The economy has been struggling
for many years but the spirit of the people remains strong. Children go to school
and dream of a better future. Parents work hard to provide food and shelter for
their families. The government has promised reforms but progress has been slow.
Many young people are looking for opportunities abroad while others choose to stay
and build their country. The weather is generally warm and sunny throughout the year.
Agriculture remains an important sector of the economy. Farmers grow maize tobacco
and cotton. The cities of Harare and Bulawayo are the largest urban centres. Many
people commute daily from surrounding towns. Public transport is often overcrowded
and unreliable. Despite these difficulties the people maintain a strong sense of
community and cultural identity. Traditional ceremonies and customs are still
widely practiced. Music and art continue to thrive as important forms of expression.
The future holds both challenges and opportunities for the nation.
"""


if __name__ == "__main__":
    # Load Shona corpus
    default_shona = "/home/compulink/Documents/z3r0 Nois3/shona-rockyou/Wordlists/cleaned/cleaned_tweets.txt"
    shona_path = sys.argv[1] if len(sys.argv) > 1 else default_shona

    if not os.path.exists(shona_path):
        print(f"[ERROR] Shona file not found: {shona_path}")
        sys.exit(1)

    with open(shona_path, "r", encoding="utf-8") as f:
        shona_text = f.read()

    # Load English text (use default or provided file)
    if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
        with open(sys.argv[2], "r", encoding="utf-8") as f:
            english_text = f.read()
    else:
        english_text = DEFAULT_ENGLISH

    run_benchmark(shona_text, english_text)
