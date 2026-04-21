"""
z3ro nois3: Adaptive Morphological Learner (V7)
------------------------------------------------
Scans any corpus, discovers the most expensive tokenization patterns,
and builds a compression dictionary automatically.

Algorithm:
  1. SCAN  — Tokenize every word via tiktoken, compute frequency
  2. RANK  — Sort by Tax Impact Score = (token_cost - 1) * frequency
  3. DISCOVER — Find recurring prefixes/suffixes via frequency analysis
  4. ALIAS  — Assign 1-token Unicode characters to top-N patterns
  5. EXPORT — Generate the compression dictionary + meta_prompt
"""

import tiktoken
from collections import Counter, defaultdict
import os
import json


class AdaptiveLearner:
    """
    Data-driven morphological pattern discoverer.
    Replaces all hardcoded dictionaries with learned patterns.
    """

    def __init__(self, model: str = "gpt-4"):
        self.enc = tiktoken.encoding_for_model(model)
        self.word_stats = {}       # {word: {tokens, frequency, impact}}
        self.morpheme_stats = {}   # {morpheme: {frequency, type, savings}}
        self.dictionary = {}       # {word_or_morpheme: alias_char}
        self.total_words = 0
        self.total_tokens = 0

        # Pool of 1-token characters validated against tiktoken.
        # We pre-build this once so alias assignment is instant.
        self._alias_pool = self._build_alias_pool()

    # ------------------------------------------------------------------
    # Phase 1: SCAN
    # ------------------------------------------------------------------
    def scan_corpus(self, text: str):
        """
        Tokenize every word and compute frequency + token cost.
        """
        words = text.split()
        self.total_words = len(words)
        clean_words = [w.lower().strip('.,!?:;\"\'\u2018\u2019\u201c\u201d()[]{}') for w in words]

        freq = Counter(clean_words)

        for word, count in freq.items():
            if len(word) < 3:
                continue
            token_cost = len(self.enc.encode(word))
            if token_cost < 2:
                continue  # Already cheap — no tax here

            impact = (token_cost - 1) * count  # How many tokens we'd save

            self.word_stats[word] = {
                "tokens": token_cost,
                "frequency": count,
                "impact": impact,
            }

        self.total_tokens = sum(
            v["tokens"] * v["frequency"] for v in self.word_stats.values()
        )

    # ------------------------------------------------------------------
    # Phase 2: DISCOVER MORPHEMES
    # ------------------------------------------------------------------
    def discover_morphemes(self, min_len: int = 3, min_freq: int = 10):
        """
        Find recurring prefixes and suffixes via frequency analysis.
        Uses a simple but effective approach: scan all words for shared
        beginnings and endings, then rank by how much tax they carry.
        """
        prefix_counter = defaultdict(int)
        suffix_counter = defaultdict(int)

        expensive_words = [
            w for w, s in self.word_stats.items() if s["tokens"] >= 3
        ]

        for word in expensive_words:
            freq = self.word_stats[word]["frequency"]
            # Extract candidate prefixes (length 3–7)
            for plen in range(min_len, min(8, len(word))):
                prefix_counter[word[:plen]] += freq
            # Extract candidate suffixes (length 3–6)
            for slen in range(min_len, min(7, len(word))):
                suffix_counter[word[-slen:]] += freq

        # Filter: morpheme must appear across multiple distinct words
        def _appears_in_n_words(morph, words, position, n=3):
            count = 0
            for w in words:
                if position == "prefix" and w.startswith(morph) and w != morph:
                    count += 1
                elif position == "suffix" and w.endswith(morph) and w != morph:
                    count += 1
                if count >= n:
                    return True
            return False

        for morph, freq in prefix_counter.items():
            if freq >= min_freq and _appears_in_n_words(morph, expensive_words, "prefix"):
                token_cost = len(self.enc.encode(morph))
                if token_cost >= 2:
                    self.morpheme_stats[morph] = {
                        "frequency": freq,
                        "type": "prefix",
                        "tokens": token_cost,
                        "savings": (token_cost - 1) * freq,
                    }

        for morph, freq in suffix_counter.items():
            if freq >= min_freq and _appears_in_n_words(morph, expensive_words, "suffix"):
                token_cost = len(self.enc.encode(morph))
                if token_cost >= 2:
                    if morph not in self.morpheme_stats:  # Don't override prefix
                        self.morpheme_stats[morph] = {
                            "frequency": freq,
                            "type": "suffix",
                            "tokens": token_cost,
                            "savings": (token_cost - 1) * freq,
                        }

    # ------------------------------------------------------------------
    # Phase 3: BUILD DICTIONARY
    # ------------------------------------------------------------------
    def build_dictionary(self, top_n: int = 150) -> dict:
        """
        Merge whole-word and morpheme candidates, rank by impact,
        assign 1-token aliases to the top N.
        """
        # Combine whole words and morphemes into one ranked list
        candidates = []

        for word, stats in self.word_stats.items():
            candidates.append({
                "pattern": word,
                "type": "word",
                "impact": stats["impact"],
                "tokens": stats["tokens"],
                "frequency": stats["frequency"],
            })

        for morph, stats in self.morpheme_stats.items():
            candidates.append({
                "pattern": morph,
                "type": stats["type"],
                "impact": stats["savings"],
                "tokens": stats["tokens"],
                "frequency": stats["frequency"],
            })

        # Sort by impact descending
        candidates.sort(key=lambda x: x["impact"], reverse=True)

        # Assign aliases (avoiding collisions)
        used_aliases = set()
        self.dictionary = {}
        alias_idx = 0

        for candidate in candidates:
            if len(self.dictionary) >= top_n:
                break
            if alias_idx >= len(self._alias_pool):
                break

            pattern = candidate["pattern"]

            # Skip if this pattern is a substring of an already-selected pattern
            # (avoid double-compression)
            skip = False
            for existing in self.dictionary:
                if pattern in existing or existing in pattern:
                    skip = True
                    break
            if skip:
                continue

            alias = self._alias_pool[alias_idx]
            self.dictionary[pattern] = {
                "alias": alias,
                "type": candidate["type"],
                "tokens": candidate["tokens"],
                "frequency": candidate["frequency"],
                "impact": candidate["impact"],
            }
            alias_idx += 1

        return self.dictionary

    # ------------------------------------------------------------------
    # Phase 4: EXPORT
    # ------------------------------------------------------------------
    def export_meta_prompt(self) -> str:
        """Auto-generate the meta_prompt.md from the learned dictionary."""
        lines = [
            "# z3ro nois3: Meta-Prompt (Auto-Generated by Adaptive Learner)",
            "",
            "## Decompression Dictionary",
            "This text uses the Bridge compression engine. Each symbol represents",
            "a high-frequency Shona word or morpheme.",
            "",
            "| Symbol | Pattern | Type | Original Tokens |",
            "| :--- | :--- | :--- | :--- |",
        ]

        for pattern, info in self.dictionary.items():
            lines.append(
                f"| `{info['alias']}` | `{pattern}` | {info['type']} | {info['tokens']} |"
            )

        lines.extend([
            "",
            "## Reconstruction Rules",
            "1. Replace each symbol with its corresponding pattern.",
            "2. Maintain semantic fidelity — do not hallucinate.",
            "",
            "---",
            "*Auto-generated by z3ro nois3 Bridge v7.0*",
        ])

        return "\n".join(lines)

    def get_compression_map(self) -> dict:
        """Return a simple {pattern: alias_char} dict for the engine."""
        return {p: info["alias"] for p, info in self.dictionary.items()}

    def get_stats_summary(self) -> dict:
        """Return a summary of the learning results."""
        total_potential_savings = sum(
            info["impact"] for info in self.dictionary.values()
        )
        return {
            "unique_words_scanned": len(self.word_stats),
            "total_words_in_corpus": self.total_words,
            "morphemes_discovered": len(self.morpheme_stats),
            "dictionary_size": len(self.dictionary),
            "total_potential_token_savings": total_potential_savings,
            "theoretical_savings_pct": (
                (total_potential_savings / self.total_tokens * 100)
                if self.total_tokens
                else 0
            ),
        }

    # ------------------------------------------------------------------
    # Internal: 1-token alias pool
    # ------------------------------------------------------------------
    def _build_alias_pool(self) -> list:
        """
        Build a list of characters that are each exactly 1 token
        in the target tokenizer. Avoids alphanumerics and common punctuation
        to prevent collisions with natural text.
        """
        pool = []

        # Latin-1 Supplement and other Unicode blocks
        safe_ranges = [
            (161, 192),   # ¡ ¢ £ ¤ ¥ ¦ § ¨ © ª « ¬ ® ¯ ° ± ² ³ ´ µ ¶ · ¸ ¹ º » ¼ ½ ¾ ¿
            (192, 256),   # À-ÿ (Latin Extended)
            (913, 940),   # Greek uppercase
            (945, 970),   # Greek lowercase
            (1025, 1120), # Cyrillic
        ]

        for start, end in safe_ranges:
            for i in range(start, end):
                c = chr(i)
                try:
                    if len(self.enc.encode(c)) == 1 and c.isprintable():
                        pool.append(c)
                except Exception:
                    pass

        return pool


if __name__ == "__main__":
    # Quick demo
    learner = AdaptiveLearner()

    corpus_path = "/home/compulink/Documents/z3r0 Nois3/shona-rockyou/Wordlists/cleaned/cleaned_tweets.txt"
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("Phase 1: Scanning corpus...")
    learner.scan_corpus(text)

    print("Phase 2: Discovering morphemes...")
    learner.discover_morphemes()

    print("Phase 3: Building dictionary (top 150)...")
    learner.build_dictionary(top_n=150)

    stats = learner.get_stats_summary()
    print(f"\n--- Learning Complete ---")
    print(f"  Words scanned     : {stats['unique_words_scanned']:,}")
    print(f"  Morphemes found   : {stats['morphemes_discovered']:,}")
    print(f"  Dictionary size   : {stats['dictionary_size']}")
    print(f"  Potential savings  : {stats['total_potential_token_savings']:,} tokens")
    print(f"  Theoretical max   : {stats['theoretical_savings_pct']:.2f}%")

    print("\nTop 10 learned patterns:")
    for i, (pat, info) in enumerate(learner.dictionary.items()):
        if i >= 10:
            break
        print(f"  {info['alias']} → {pat} ({info['type']}, {info['tokens']} tok, impact={info['impact']})")
