# z3r0 Noise Project Overview

This repository contains tools and scripts for deep linguistic analysis of the Shona language, calculating algorithmic complexity, and analyzing the "Bantu Tax" incurred by Western-centric NLP models (like BPE tokenizers). 

## How to Run

### Python CLI Scripts
Run these directly via the terminal. They generally require a text corpus as an argument.
```bash
python3 shona_corpus_analysis.py <corpus_file.txt>
python3 shona_pitch_engine.py <shona_corpus.txt>
python3 shona_leet_mutator.py
```

### Streamlit Dashboards
Run these interactive dashboards using Streamlit to view them in your web browser.
```bash
streamlit run dashboard.py
streamlit run shona_leet_mutator2.py
```

## File Purposes & Comparison Table

| File | Primary Purpose | Interface | Execution Flow |
|------|----------------|-----------|----------------|
| `dashboard.py` | Analyzes and visually compares the tokenization costs and efficiencies (Bantu Tax) between English and Shona text models. | Streamlit UI | Requires file uploads (English/Shona text) directly through the browser. |
| `shona_corpus_analysis.py` | The main linguistic engine for deep structural, statistical, phonotactic, and entropy analysis of an input Shona text corpus. | Command Line | Takes a single corpus text file. Outputs analytical charts (`.png` files) and console tables. |
| `shona_leet_mutator.py` | Core algorithmic tool to generate "leet-speak" password mutations for Shona words, establishing complexity baselines and entropy differences. | Command Line | Automatically runs against a pre-defined local file (`names_surnames.txt`). |
| `shona_leet_mutator2.py` | The interactive dashboard version of `shona_leet_mutator.py` to visually demonstrate the security search space concepts. | Streamlit UI | Automatically reads `names_given.txt` upon running. |
| `shona_pitch_engine.py` | A summarized implementation of the engine highlighting 3 key pitch features: predictability analysis, noise refinery, and slot-aware token benchmarking. | Command Line | Takes a Shona corpus `.txt`. Used for high-level demonstrations and validation metrics. |
| `Genesis_full.txt` & `dummy_english.txt` | Data corporas used as inputs for testing the analyzers. | Data File | N/A |
| `Pycon Talk Outline.md` | Talk outline and structure describing the motivation and technical background of the project for a conference pitch. | Document | N/A |
