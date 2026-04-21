"""
z3ro nois3: Bridge Engine (V7 - Adaptive)
-----------------------------------------
No more hardcoded dictionaries. The engine learns from any corpus
using the AdaptiveLearner, then applies the discovered patterns.
"""

import re
from learner import AdaptiveLearner


class BantuBridgeEngine:
    """
    Adaptive compression engine for agglutinative languages.
    
    Usage:
        engine = BantuBridgeEngine()
        engine.learn("path/to/corpus.txt", top_n=150)
        optimized = engine.process_text("Vachikumbira zvakanaka")
    """

    def __init__(self):
        self.learner = AdaptiveLearner()
        self.compression_map = {}     # {pattern: alias_char}
        self.sorted_patterns = []     # Sorted longest-first for greedy matching
        self.is_trained = False
        self.stats = {}

    def learn(self, corpus_text: str, top_n: int = 150):
        """
        Train the engine on a corpus. This replaces hardcoded dictionaries.
        
        Args:
            corpus_text: Raw text content of the corpus
            top_n: Number of patterns to learn (default 150)
        """
        self.learner.scan_corpus(corpus_text)
        self.learner.discover_morphemes()
        self.learner.build_dictionary(top_n=top_n)

        self.compression_map = self.learner.get_compression_map()
        self.sorted_patterns = sorted(
            self.compression_map.keys(), key=len, reverse=True
        )
        self.stats = self.learner.get_stats_summary()
        self.is_trained = True

    def learn_from_file(self, filepath: str, top_n: int = 150):
        """Convenience method to learn from a file path."""
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        self.learn(text, top_n=top_n)

    def optimize_word(self, word: str) -> str:
        """
        Apply learned compression to a single word.
        Checks for whole-word match first (highest savings),
        then tries prefix/suffix morpheme matches.
        """
        if not self.is_trained or len(word) < 3:
            return word

        clean = word.lower().strip('.,!?:;\"\'\u2018\u2019\u201c\u201d()[]{}')

        # Tier 1: Exact whole-word match
        if clean in self.compression_map:
            return self.compression_map[clean]

        # Tier 2: Morpheme match (prefix then suffix)
        best_result = word
        best_savings = 0

        for pattern in self.sorted_patterns:
            info = self.learner.dictionary.get(pattern, {})
            ptype = info.get("type", "word")

            if ptype == "prefix" and clean.startswith(pattern) and len(clean) > len(pattern):
                alias = self.compression_map[pattern]
                candidate = alias + clean[len(pattern):]
                savings = len(pattern) - len(alias)
                if savings > best_savings:
                    best_result = candidate
                    best_savings = savings

            elif ptype == "suffix" and clean.endswith(pattern) and len(clean) > len(pattern):
                alias = self.compression_map[pattern]
                candidate = clean[: -len(pattern)] + alias
                savings = len(pattern) - len(alias)
                if savings > best_savings:
                    best_result = candidate
                    best_savings = savings

        return best_result

    def process_text(self, text: str) -> str:
        """Process a full text string word-by-word."""
        words = text.split()
        return " ".join(self.optimize_word(w) for w in words)

    def get_meta_prompt(self) -> str:
        """Get the auto-generated meta prompt for LLM decompression."""
        if not self.is_trained:
            return "# Engine not trained yet. Call engine.learn() first."
        return self.learner.export_meta_prompt()

    def get_training_stats(self) -> dict:
        """Return training statistics."""
        return self.stats


if __name__ == "__main__":
    engine = BantuBridgeEngine()

    corpus_path = "/home/compulink/Documents/z3r0 Nois3/shona-rockyou/Wordlists/cleaned/cleaned_tweets.txt"
    print("Training engine on tweet corpus...")
    engine.learn_from_file(corpus_path, top_n=150)

    stats = engine.get_training_stats()
    print(f"  Dictionary size: {stats['dictionary_size']}")
    print(f"  Potential savings: {stats['theoretical_savings_pct']:.2f}%\n")

    test_sentences = [
        "Vachikumbira zvakanaka muChitungwiza",
        "Vanhu varikuda kudya chikafu chakanaka",
        "Hatingavaka patsva Zimbabwe pasina kutenda",
    ]
    for s in test_sentences:
        opt = engine.process_text(s)
        print(f"  IN:  {s}")
        print(f"  OUT: {opt}\n")
