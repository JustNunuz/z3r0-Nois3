
# Z3RO NOIS3

![Z3ro Nois3 Header](z3r0-nois3/img/Gemini_Generated_Image_ki8oduki8oduki8o.png)

## 🌍 The Mission
**Z3ro Nois3** is a linguistic forensic engine designed to audit the structural and economic exclusion of Bantu languages in modern Large Language Models (LLMs).

This monorepo brings together two projects:

| Module | Description |
| :--- | :--- |
| [`z3r0-nois3/`](z3r0-nois3/) | **The Engine** — CLI & Streamlit tools that audit tokenization costs, linguistic entropy, security search spaces, and morphological forensics. |
| [`shona-rockyou/`](shona-rockyou/) | **The Corpus** — A research-oriented Shona linguistic dataset of names, surnames, totems, place names, and morphological components. Linked as a [git submodule](https://github.com/JustNunuz/Shona-Rockyou). |

## 🚀 Quick Start

### 1. Clone (with submodule data)
```bash
git clone --recurse-submodules https://github.com/JustNunuz/z3r0-Noise.git
cd z3r0-Noise
```

If you already cloned without `--recurse-submodules`:
```bash
git submodule update --init --recursive
```

### 2. Install Dependencies
```bash
pip install -r z3r0-nois3/requirements.txt
```

### 3. Run the Engine
```bash
# CLI analysis
cd z3r0-nois3
python3 shona_corpus_analysis.py Genesis_full.txt

# Streamlit dashboards
streamlit run dashboard.py
streamlit run shona_leet_mutator2.py
```

## 📂 Project Structure

```
z3r0 Nois3/
├── README.md                    ← you are here (monorepo root)
│
├── z3r0-nois3/                  ← the forensic engine
│   ├── README.md                ← detailed engine docs
│   ├── requirements.txt
│   ├── dashboard.py             ← Bantu Tax tokenization dashboard
│   ├── shona_corpus_analysis.py ← deep linguistic analysis engine
│   ├── shona_leet_mutator.py    ← password mutation CLI
│   ├── shona_leet_mutator2.py   ← password mutation Streamlit UI
│   ├── shona_pitch_engine.py    ← pitch feature engine
│   ├── Pycon Talk Outline.md
│   ├── Genesis_full.txt
│   ├── dummy_english.txt
│   └── img/
│
└── shona-rockyou/               ← git submodule (corpus data)
    ├── README.md
    ├── Wordlists/
    │   ├── names_given.txt
    │   ├── names_surnames.txt
    │   ├── names_given_mutations.txt
    │   ├── names_surnames_mutations.txt
    │   ├── totems.txt
    │   ├── geography.txt
    │   ├── popculture.txt
    │   └── tweets.txt
    └── Analysis/
        ├── readme.md
        ├── rockyou_analysis.md
        ├── NGM_readme.txt
        └── NSM_readme.txt
```

## 🔗 How the Modules Work Together

The **Shona-Rockyou** corpus provides the raw linguistic seeds (names, totems, places) that the **z3r0-nois3** engine consumes to:

1. **Audit the "Bantu Tax"** — quantify the cost difference of tokenizing Shona vs English.
2. **Generate Security Mutations** — produce leet-speak password variants and measure entropy delta.
3. **Map Sub-word Glitches** — visualize where AI tokenizers mangle meaningful Shona morphemes.
4. **Benchmark Slot-Based Tokenization** — propose more efficient processing for agglutinative languages.

## 🛡️ Security & DPO Perspective
As a project led by a **Certified Data Protection Officer (DPO)**, Z3ro Nois3 explores how linguistic bias affects data privacy. When an AI doesn't understand the morphology of a language, it cannot properly audit the entropy of its users' data, leading to a "False Security" trap in the SADC region.

## 🤝 Call for Contributors
Help us bridge the linguistic gap! We are expanding our engine to support a wider range of Bantu languages.

* **Language Experts:** Help us map the morphological "slots" for Ndebele, Zulu, Swahili, and more.
* **Data Providers:** We are looking for clean, diverse datasets to improve our forensic audits.
* **Developers:** Help us optimize the Python middleware for regional scale.

*Built with ☕ for the future of African AI.*
