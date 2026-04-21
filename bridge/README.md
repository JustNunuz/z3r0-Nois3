# Z3r0 Nois3: Tokenization Bridge

## Overview
The **Bridge** module represents the experimental machine learning and adaptive engine of the broader Z3r0 Nois3 project. Rather than relying on static dictionaries or forced Western BPE implementations, the Bridge physically implements and tests alternative approaches to mapping Bantu languages for AI.

## Conceptual Structure
Instead of a collection of disconnected scripts, the bridge functions as an iterative pipeline:

* **Tokenization Learning Engine:** The core system components (like `learner.py` and `engine.py`) that ingest cleaned datasets from the `shona-rockyou` corpus and iteratively train an **unsupervised, rule-discovering tokenizer**. This is where the code dynamically identifies Shona morphological boundaries and adapts to structure without being told "how" to read the language.
* **Benchmarking & Analysis Suite:** Tools used to programmatically pit standard tokenizers (like ordinary BPE) against our custom patterns. These files rigorously quantify metrics like token length, fertility, subword glitches, and economic cost (the "Bantu Tax").
* **Interactive APIs:** Web servers and Streamlit applications (such as `app.py`) designed to visualize the progress of the learning model and enable researchers to intuitively query the tokenizer over HTTP.

## The Goal
The Bridge takes the static proof created by the rest of the project and builds a **dynamic solution**. By automatically learning morphological slots, the Bridge acts as the connecting layer that can sit seamlessly between raw African language structures and massive GenAI language models.
