"""
z3ro nois3: Bridge CLI Benchmark (V7 - Adaptive)
-------------------------------------------------
Runs the adaptive engine on a corpus and reports compression metrics.
"""

import tiktoken
import sys
import os
from engine import BantuBridgeEngine


def run_benchmark(filepath, dict_size=150):
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return

    print(f"{'='*70}")
    print(f" z3ro nois3: Bridge PoC Benchmark (V7 Adaptive)")
    print(f" Source: {os.path.basename(filepath)}")
    print(f" Dictionary Size: {dict_size}")
    print(f"{'='*70}\n")

    # Load corpus
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # Train the adaptive engine
    print("  Phase 1: Learning patterns from corpus...")
    engine = BantuBridgeEngine()
    engine.learn(text, top_n=dict_size)
    stats = engine.get_training_stats()

    print(f"  ├─ Words scanned    : {stats['unique_words_scanned']:,}")
    print(f"  ├─ Morphemes found  : {stats['morphemes_discovered']:,}")
    print(f"  ├─ Dictionary built : {stats['dictionary_size']} entries")
    print(f"  └─ Theoretical max  : {stats['theoretical_savings_pct']:.2f}%\n")

    # Benchmark
    print("  Phase 2: Compressing corpus...\n")
    enc = tiktoken.encoding_for_model("gpt-4")

    total_orig = 0
    total_opt = 0
    count = 0
    optimized_count = 0
    pure_savings = 0
    pure_words = 0

    for line in lines:
        t_orig = len(enc.encode(line))
        opt_line = engine.process_text(line)
        t_opt = len(enc.encode(opt_line))

        total_orig += t_orig
        total_opt += t_opt
        count += 1

        was_optimized = False
        for w, ow in zip(line.split(), opt_line.split()):
            if w != ow:
                was_optimized = True
                pure_words += 1
                pure_savings += (len(enc.encode(w)) - len(enc.encode(ow)))

        if was_optimized:
            optimized_count += 1

        if count <= 10:
            if was_optimized:
                print(f"  [OPT] {line[:35]:35} -> {opt_line[:35]:35} | -{t_orig-t_opt}")
            else:
                print(f"        {line[:35]:35} (no match)")

    if count > 10:
        print(f"\n  ... and {count-10:,} more entries.\n")

    total_saved = total_orig - total_opt
    savings_pct = (total_saved / total_orig * 100) if total_orig else 0
    coverage_pct = (optimized_count / count * 100) if count else 0

    print(f"{'─'*70}")
    print(f" FINAL METRICS (Adaptive V7)")
    print(f"{'─'*70}")
    print(f" Total Entries           : {count:,}")
    print(f" Linguistic Coverage     : {coverage_pct:.2f}%")
    print(f" Total Baseline Tokens   : {total_orig:,}")
    print(f" Total Bridge Tokens     : {total_opt:,}")
    print(f" Total Tokens Saved      : {total_saved:,}")
    print(f" Overall Savings         : {savings_pct:.2f}%")
    print(f"\n STRUCTURAL IMPACT")
    print(f"{'─'*70}")
    print(f" Words Optimized         : {pure_words:,}")
    print(f" Avg Saving per Match    : {pure_savings/pure_words if pure_words else 0:.2f} tokens")
    print(f" Cost Saved (per run)    : ${total_saved * 0.03 / 1000:.4f}")
    print(f" Cost Saved (per 1M tok) : ${total_saved * 0.03 / 1000 * 1000000 / total_orig:.2f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    default_file = "/home/compulink/Documents/z3r0 Nois3/shona-rockyou/Wordlists/cleaned/cleaned_tweets.txt"
    target = sys.argv[1] if len(sys.argv) > 1 else default_file
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    run_benchmark(target, dict_size=size)
