"""
z3ro nois3: Bridge Dashboard (V7 - Adaptive)
---------------------------------------------
Features:
  - Adaptive learning from uploaded corpus
  - Multi-tokenizer comparison (BPE, WordPiece, SentencePiece, Bridge)
  - Cost analysis and fertility metrics
  - Academic citations for the Bantu Tax
"""

import streamlit as st
import pandas as pd
import tiktoken
from engine import BantuBridgeEngine
from tokenizer_bench import (
    bpe_tokenize, wordpiece_tokenize,
    sentencepiece_tokenize, count_words
)

# -- Page Config --
st.set_page_config(page_title="z3ro nois3 | bridge v7", layout="wide", page_icon="🦖")

st.markdown("""
<style>
.main { background-color: #0d1117; color: #e6edf3; }
.stMetric { background-color: #161b22; padding: 15px; border-radius: 10px;
            border: 1px solid #30363d; }
.stDataFrame { border: 1px solid #30363d; }
h1,h2,h3 { color: #e6edf3; }
</style>
""", unsafe_allow_html=True)

# -- Header --
st.title("🦖 z3ro nois3: The bridge (V7)")
st.subheader("Adaptive Morphological Compression Engine")

# -- Sidebar --
st.sidebar.title("⚙️ Configuration")
model_selector = st.sidebar.selectbox("Baseline Model", ["gpt-4", "gpt-3.5-turbo"])
cost_per_1k = st.sidebar.number_input("Cost per 1K Tokens ($)", value=0.03, format="%.4f")
dict_size = st.sidebar.slider("Dictionary Size (top-N)", 50, 300, 150)

# -- Tabs --
tab_bridge, tab_bench, tab_research = st.tabs([
    "🏗️ Bridge Compression", "📊 Tokenizer Comparison", "📚 The Bantu Tax (Research)"
])

# ══════════════════════════════════════════════════════════════════
# TAB 1: Bridge Compression (Adaptive)
# ══════════════════════════════════════════════════════════════════
with tab_bridge:
    st.write("### 1. Upload a Shona Corpus")
    st.caption("The engine will **learn** patterns from your text, then compress it.")
    uploaded = st.file_uploader("Upload .txt file", type=["txt"], key="bridge_upload")

    if uploaded:
        raw = uploaded.read().decode("utf-8")
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        corpus_text = "\n".join(lines)

        st.success(f"Loaded {len(lines):,} lines from **{uploaded.name}**")

        # Train the adaptive engine
        with st.spinner("🧠 Learning patterns from corpus..."):
            engine = BantuBridgeEngine()
            engine.learn(corpus_text, top_n=dict_size)
            stats = engine.get_training_stats()

        # Show learning stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Words Scanned", f"{stats['unique_words_scanned']:,}")
        with col2:
            st.metric("Morphemes Found", f"{stats['morphemes_discovered']:,}")
        with col3:
            st.metric("Dictionary Size", stats['dictionary_size'])
        with col4:
            st.metric("Theoretical Max", f"{stats['theoretical_savings_pct']:.1f}%")

        st.markdown("---")

        # Compress and measure
        enc = tiktoken.encoding_for_model(model_selector)
        results = []
        total_orig = 0
        total_opt = 0

        for line in lines:
            t_orig = len(enc.encode(line))
            opt = engine.process_text(line)
            t_opt = len(enc.encode(opt))
            results.append({
                "Original": line[:80],
                "Compressed": opt[:80],
                "Tokens (Orig)": t_orig,
                "Tokens (Opt)": t_opt,
                "Saved": t_orig - t_opt,
            })
            total_orig += t_orig
            total_opt += t_opt

        savings_pct = ((total_orig - total_opt) / total_orig * 100) if total_orig else 0
        tokens_saved = total_orig - total_opt
        cost_saved = tokens_saved * cost_per_1k / 1000

        st.write("### 2. Compression Results")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Baseline Tokens", f"{total_orig:,}")
        with c2:
            st.metric("Bridge Tokens", f"{total_opt:,}", delta=f"-{tokens_saved:,}")
        with c3:
            st.metric("Tax Reduction", f"{savings_pct:.2f}%")
        with c4:
            st.metric("Cost Saved", f"${cost_saved:.4f}")

        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        # Export
        st.write("### 3. Export")
        col_a, col_b = st.columns(2)
        with col_a:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results (CSV)", csv, "bridge_results.csv", "text/csv")
        with col_b:
            meta = engine.get_meta_prompt()
            st.download_button("Download Meta-Prompt", meta.encode("utf-8"),
                               "meta_prompt.md", "text/markdown")
    else:
        st.info("Upload a .txt file to begin adaptive learning.")

        with st.expander("How does the Adaptive Engine work?"):
            st.markdown("""
1. **SCAN** — Tokenize every word via `tiktoken` and compute frequency
2. **RANK** — Sort words by *Tax Impact Score* = `(token_cost - 1) × frequency`
3. **DISCOVER** — Find recurring prefixes/suffixes via frequency analysis
4. **ALIAS** — Assign 1-token Unicode characters to the top-N patterns
5. **COMPRESS** — Apply the learned dictionary to reduce token count
            """)

# ══════════════════════════════════════════════════════════════════
# TAB 2: Tokenizer Comparison
# ══════════════════════════════════════════════════════════════════
with tab_bench:
    st.write("### Tokenizer Forensics: English vs Shona")
    st.caption("Compare BPE (GPT-4), WordPiece (BERT), SentencePiece (Google), and Bridge.")

    col_en, col_sn = st.columns(2)

    with col_en:
        st.write("**English Text**")
        en_upload = st.file_uploader("Upload English .txt", type=["txt"], key="en_upload")
        en_default = (
            "The people of Zimbabwe are hardworking and resilient. "
            "They wake up every morning and go to work despite the challenges they face. "
            "The economy has been struggling for many years but the spirit of the people "
            "remains strong. Children go to school and dream of a better future. "
            "Parents work hard to provide food and shelter for their families."
        )
        en_text = en_upload.read().decode("utf-8") if en_upload else en_default

    with col_sn:
        st.write("**Shona Text**")
        sn_upload = st.file_uploader("Upload Shona .txt", type=["txt"], key="sn_upload")

    if sn_upload or st.button("Run with default Shona corpus"):
        if sn_upload:
            sn_text = sn_upload.read().decode("utf-8")
        else:
            default_path = "/home/compulink/Documents/z3r0 Nois3/shona-rockyou/Wordlists/cleaned/cleaned_tweets.txt"
            import os
            if os.path.exists(default_path):
                with open(default_path, "r", encoding="utf-8") as f:
                    sn_text = f.read()
            else:
                st.error("Default Shona corpus not found.")
                st.stop()

        en_words = count_words(en_text)
        sn_words = count_words(sn_text)

        with st.spinner("Running tokenizer benchmark (this may take 1-2 minutes)..."):
            # English
            en_bpe = bpe_tokenize(en_text)
            en_wp = wordpiece_tokenize(en_text)
            en_sp = sentencepiece_tokenize(en_text)

            # Shona
            sn_bpe = bpe_tokenize(sn_text)
            sn_wp = wordpiece_tokenize(sn_text)
            sn_sp = sentencepiece_tokenize(sn_text)

            # Bridge
            engine = BantuBridgeEngine()
            engine.learn(sn_text, top_n=dict_size)
            opt_sn = engine.process_text(sn_text)
            sn_bridge = bpe_tokenize(opt_sn)

        en_bpe_f = en_bpe / en_words if en_words else 1

        rows = [
            {"Tokenizer": "BPE (GPT-4)", "Language": "English",
             "Tokens": en_bpe, "Fertility": en_bpe / en_words,
             "Cost ($)": en_bpe * cost_per_1k / 1000, "Tax vs EN BPE": 1.0},
            {"Tokenizer": "WordPiece", "Language": "English",
             "Tokens": en_wp, "Fertility": en_wp / en_words,
             "Cost ($)": en_wp * cost_per_1k / 1000,
             "Tax vs EN BPE": (en_wp / en_words) / en_bpe_f},
            {"Tokenizer": "SentencePiece", "Language": "English",
             "Tokens": en_sp, "Fertility": en_sp / en_words,
             "Cost ($)": en_sp * cost_per_1k / 1000,
             "Tax vs EN BPE": (en_sp / en_words) / en_bpe_f},
            {"Tokenizer": "BPE (GPT-4)", "Language": "Shona",
             "Tokens": sn_bpe, "Fertility": sn_bpe / sn_words,
             "Cost ($)": sn_bpe * cost_per_1k / 1000,
             "Tax vs EN BPE": (sn_bpe / sn_words) / en_bpe_f},
            {"Tokenizer": "WordPiece", "Language": "Shona",
             "Tokens": sn_wp, "Fertility": sn_wp / sn_words,
             "Cost ($)": sn_wp * cost_per_1k / 1000,
             "Tax vs EN BPE": (sn_wp / sn_words) / en_bpe_f},
            {"Tokenizer": "SentencePiece", "Language": "Shona",
             "Tokens": sn_sp, "Fertility": sn_sp / sn_words,
             "Cost ($)": sn_sp * cost_per_1k / 1000,
             "Tax vs EN BPE": (sn_sp / sn_words) / en_bpe_f},
            {"Tokenizer": "Bridge + BPE", "Language": "Shona",
             "Tokens": sn_bridge, "Fertility": sn_bridge / sn_words,
             "Cost ($)": sn_bridge * cost_per_1k / 1000,
             "Tax vs EN BPE": (sn_bridge / sn_words) / en_bpe_f},
        ]

        bench_df = pd.DataFrame(rows)
        bench_df["Fertility"] = bench_df["Fertility"].round(2)
        bench_df["Tax vs EN BPE"] = bench_df["Tax vs EN BPE"].round(2)
        bench_df["Cost ($)"] = bench_df["Cost ($)"].round(4)

        import plotly.express as px

        st.write("### Results")
        st.dataframe(bench_df, use_container_width=True)

        st.write("---")
        st.write("### 📈 Visual Comparisons")
        
        col_f, col_t = st.columns(2)
        
        with col_f:
            st.write("**Fertility (Tokens per Word)**")
            st.caption("Lower is better. Compare Shona and English side-by-side.")
            
            fig_f = px.bar(
                bench_df, x="Tokenizer", y="Fertility", color="Language", 
                barmode="group", color_discrete_sequence=["#2ca02c", "#d62728"]
            )
            fig_f.update_layout(margin=dict(l=0, r=0, t=20, b=0), xaxis_title="Algorithm")
            st.plotly_chart(fig_f, use_container_width=True)

        with col_t:
            st.write("**The Bantu Tax (Relative Cost vs EN BPE)**")
            st.caption("English BPE is exactly 1.0x.")
            
            fig_t = px.bar(
                bench_df, x="Tokenizer", y="Tax vs EN BPE", color="Language", 
                barmode="group", color_discrete_sequence=["#2ca02c", "#d62728"]
            )
            fig_t.update_layout(margin=dict(l=0, r=0, t=20, b=0), xaxis_title="Algorithm")
            st.plotly_chart(fig_t, use_container_width=True)

        st.write("---")

        # Bridge impact
        bridge_saved = sn_bpe - sn_bridge
        bridge_pct = (bridge_saved / sn_bpe * 100) if sn_bpe else 0
        st.success(
            f"**Bridge Impact:** {bridge_saved:,} tokens saved "
            f"({bridge_pct:.2f}%) = ${bridge_saved * cost_per_1k / 1000:.4f} saved per run"
        )

# ══════════════════════════════════════════════════════════════════
# TAB 3: Research & Citations
# ══════════════════════════════════════════════════════════════════
with tab_research:
    st.write("### 📚 The Bantu Tax: Academic Evidence")

    st.markdown("""
The **"Bantu Tax"** is a measurable, documented phenomenon in peer-reviewed NLP research.
It refers to the economic and performance penalty imposed on speakers of agglutinative
African languages by Western-optimized AI tokenizers.

---

#### Key Papers

| Paper | Venue | Finding |
| :--- | :--- | :--- |
| **Ahia et al. (2023)** "Do All Languages Cost the Same?" | **EMNLP 2023** | Speakers of low-resource languages pay **up to 15× more** per semantic unit via API pricing due to tokenizer fertility bias. |
| **"The Token Tax"** (2026) | **ACL Anthology** | Higher fertility in African languages **directly predicts lower model accuracy** on AfriMMLU benchmarks. |
| **Petrov et al.** "Language Model Tokenization Disparity" | **arXiv** | Fertility creates a structural barrier: higher fertility → more context window consumed → worse reasoning quality. |

---

#### The Core Metric: Fertility Ratio

**Fertility** = `tokens / words`

| Language | Fertility (BPE) | Cost per 1K semantic units | Tax vs English |
| :--- | :--- | :--- | :--- |
| English | ~1.2 | $0.036 | 1.0× |
| Shona | ~2.5–3.5 | $0.075–$0.105 | **2.1–2.9×** |
| French | ~1.4 | $0.042 | 1.2× |
| Finnish | ~2.2 | $0.066 | 1.8× |

---

#### What This Means

1. **Economic Exclusion:** A Shona speaker pays **2–3× more** than an English speaker
   for the same AI capability.
2. **Performance Degradation:** A 4096-token context window holds ~3400 English words
   but only ~1200–1600 Shona words. Shona speakers hit the "memory wall" 2–3× faster.
3. **Structural Bias:** This is not a data problem — it's an **architecture** problem.
   Adding more Shona training data to BPE will not fix the fertility ratio because
   BPE is frequency-based and will always favor the dominant language.

---

#### The Bridge Solution

The z3ro nois3 Bridge is a **pre-tokenization middleware** that compresses Shona text
before it reaches the tokenizer, reducing the fertility ratio without requiring any
changes to the underlying LLM architecture.

| Approach | Requires Retraining? | Savings | Limitation |
| :--- | :--- | :--- | :--- |
| Bridge (Adaptive) | ❌ No | 5–10% global | Bounded by vocabulary diversity |
| Custom BPE Tokenizer | ✅ Full retrain | 40–60% | Requires massive compute |
| Native Morphological Tokenizer | ✅ New architecture | 60%+ | Does not exist yet |
    """)

st.markdown("---")
st.caption("z3ro nois3 Bridge V7 | Adaptive Morphological Compression | Built for Linguistic Sovereignty")
