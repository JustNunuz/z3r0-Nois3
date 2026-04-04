
# Z3RO NOIS3

![Z3ro Nois3 Header](img/Gemini_Generated_Image_ki8oduki8oduki8o.png)

## 🌍 The Mission
**Z3ro Nois3** is a linguistic forensic engine designed to audit the structural and economic exclusion of Bantu languages in modern Large Language Models (LLMs). 

In the current AI landscape, African languages are often "taxed" by inefficient, Western-centric tokenizers (like BPE). This project provides the data-driven proof and the technical middleware to reclaim **Linguistic Sovereignty** ensuring that AI works as efficiently for a Shona speaker in Harare as it does for an English speaker in San Francisco.


## 🚀 Key Features
* **The "Bantu Tax" Auditor:** Quantifies the real-world cost difference between processing English and Shona text.
* **Morpheme-Aware Forensics:** Tools to visualize "Sub-word Glitches" where AI mangles meaningful Shona roots.
* **Cultural Entropy Engine:** A security-focused analyzer proving why standard password strength meters fail to account for Bantu linguistic patterns.
* **Slot-Based Token Benchmarking:** Proposing a new efficiency standard for agglutinative language processing.

## 🛠️ How to Run

### **Python CLI Scripts**
Run these directly via the terminal. They generally require a text corpus as an argument.
```bash
python3 shona_corpus_analysis.py <corpus_file.txt>
python3 shona_pitch_engine.py <shona_corpus.txt>
python3 shona_leet_mutator.py
```

### **Streamlit Dashboards**
Run these interactive dashboards using Streamlit to view them in your web browser.
```bash
streamlit run dashboard.py
streamlit run shona_leet_mutator2.py
```

## 📂 File Purposes & Comparison Table

| File | Primary Purpose | Interface | Execution Flow |
| :--- | :--- | :--- | :--- |
| `dashboard.py` | Analyzes and visually compares the tokenization costs and efficiencies (Bantu Tax) between English and Shona text models. | Streamlit UI | Requires file uploads (English/Shona text) directly through the browser. |
| `shona_corpus_analysis.py` | The main linguistic engine for deep structural, statistical, phonotactic, and entropy analysis of an input Shona text corpus. | Command Line | Takes a single corpus text file. Outputs analytical charts (`.png` files) and console tables. |
| `shona_leet_mutator.py` | Core algorithmic tool to generate "leet-speak" password mutations for Shona words, establishing complexity baselines and entropy differences. | Command Line | Automatically runs against a pre-defined local file (`names_surnames.txt`). |
| `shona_leet_mutator2.py` | The interactive dashboard version of `shona_leet_mutator.py` to visually demonstrate the security search space concepts. | Streamlit UI | Automatically reads `names_given.txt` upon running. |
| `shona_pitch_engine.py` | A summarized implementation of the engine highlighting 3 key pitch features: predictability analysis, noise refinery, and slot-aware token benchmarking. | Command Line | Takes a Shona corpus `.txt`. Used for high-level demonstrations and validation metrics. |
| `Genesis_full.txt` & `dummy_english.txt` | Data corporas used as inputs for testing the analyzers. | Data File | N/A |
| `Pycon Talk Outline.md` | Talk outline and structure describing the motivation and technical background of the project for a conference pitch. | Document | N/A |


## 🛡️ Security & DPO Perspective
As a project led by a **Certified Data Protection Officer (DPO)**, Z3ro Nois3 explores how linguistic bias affects data privacy. When an AI doesn't understand the morphology of a language, it cannot properly audit the entropy of its users' data, leading to a "False Security" trap in the SADC region.

## 🤝 Call for Contributors:
Help us bridge the linguistic gap! We are expanding our engine to support a wider range of Bantu languages and we need your help.

* Language Experts: Help us map the morphological "slots" for Ndebele, Zulu, Swahili, and more.

* Data Providers: We are looking for clean, diverse datasets to improve our forensic audits.

* Developers: Help us optimize the Python middleware for regional scale.

*Built with ☕ for the future of African AI.*
