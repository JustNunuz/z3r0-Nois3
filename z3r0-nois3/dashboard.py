import streamlit as st
import tiktoken
import pandas as pd
import io
import time

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Tokenization Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark Security Dashboard Theme with Monospace via Markdown:
st.markdown("""
<style>
    /* Global Monospace & Colors */
    html, body, [class*="css"] {
        font-family: 'Courier New', Courier, monospace !important;
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Security Dashboard Specific Styling */
    h1, h2, h3, h4 {
        color: #58a6ff !important;
        border-bottom: 1px solid #30363d;
        padding-bottom: 0.3em;
    }
    .stMetric > div > div > div {
        color: #3fb950 !important;
    }
    .stMetric label {
        color: #8b949e !important;
    }
    
    /* Enhance Tables */
    .stDataFrame {
        border: 1px solid #30363d;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Bantu Tax Tokenization Dashboard")
st.markdown("Analyzing the severe efficiency gaps in English vs. African Languages (Shona) using BPE Models.")

# --- EDITABLE PRICING SECTION (Update as per April 2026 OpenAI rates) ---
PRICE_PER_1M_TOKENS = 10.00  # Default: GPT-4 Turbo Input Rate ($)
# -----------------------------------------------------------------------

def calculate_cost(token_count):
    """Calculates the cost based on token count."""
    return (token_count / 1_000_000) * PRICE_PER_1M_TOKENS

@st.cache_resource
def get_tokenizers():
    return tiktoken.get_encoding("cl100k_base"), tiktoken.get_encoding("o200k_base")

enc_cl100k, enc_o200k = get_tokenizers()

def analyze_text(text, name, tokenizer):
    words = text.split()
    total_words = len(words)
    total_chars = len(text)
    tokens = tokenizer.encode(text)
    total_tokens = len(tokens)
    
    fertility_ratio = total_tokens / total_words if total_words > 0 else 0
    chars_per_token = total_chars / total_tokens if total_tokens > 0 else 0
    cost = calculate_cost(total_tokens)
    
    glitches = 0
    fragments_list = []
    worst_offenders = []
    
    for word in words:
        word_tokens = tokenizer.encode(word)
        num_word_tokens = len(word_tokens)
        fragments_list.append(num_word_tokens)
        if num_word_tokens > 4:
            glitches += 1
            worst_offenders.append({
                "Word": word,
                "Length": len(word),
                "Tokens": num_word_tokens,
                "Fragments": " | ".join([tokenizer.decode([t]) for t in word_tokens])
            })
            
    avg_fragments = sum(fragments_list) / len(fragments_list) if len(fragments_list) > 0 else 0
    
    # Sort worst offenders by token fragmentation
    worst_offenders = sorted(worst_offenders, key=lambda x: x["Tokens"], reverse=True)[:10]
    
    return {
        "Dataset": name,
        "Total Words": total_words,
        "Total Characters": total_chars,
        "Total Tokens": total_tokens,
        "Fertility Ratio": round(fertility_ratio, 3),
        "Char/Token Density": round(chars_per_token, 3),
        "Cost ($)": cost,
        "Deep Fragmentation Glitches": glitches,
        "Avg Fragments/Word": round(avg_fragments, 3),
        "Worst Offenders": worst_offenders
    }

st.header("1. Upload Corpora")
col_eng, col_shn = st.columns(2)
with col_eng:
    eng_file = st.file_uploader("Upload English Text (upload_english.txt)", type=["txt"])
with col_shn:
    shn_file = st.file_uploader("Upload Shona Text (upload_shona.txt)", type=["txt"])

if eng_file and shn_file:
    # Use session state to avoid re-triggering the 15s wait on every input change (like the forensic text box)
    current_file_id = f"{eng_file.name}_{eng_file.size}_{shn_file.name}_{shn_file.size}"
    
    if st.session_state.get("last_uploaded") != current_file_id:
        status_placeholder = st.empty()
        messages = [
            "Initializing Tokenizers...",
            "Decoding text streams...",
            "Calculating Fertility Ratios...",
            "Analyzing Deep Fragmentation...",
            "Mapping Sub-word Glitches...",
            "Finalizing Analytics Matrix..."
        ]
        delay = 15.0 / len(messages)
        for msg in messages:
            status_placeholder.warning(f"⏳ **Loading Data:** {msg}")
            time.sleep(delay)
        status_placeholder.empty()
        st.session_state["last_uploaded"] = current_file_id

    # Decode files
    eng_text = eng_file.read().decode("utf-8")
    shn_text = shn_file.read().decode("utf-8")
    
    # Process using cl100k_base by default for the primary metric
    eng_metrics = analyze_text(eng_text, "English", enc_cl100k)
    shn_metrics = analyze_text(shn_text, "Shona", enc_cl100k)
    
    # Section 2: Metrics
    st.header("2. Primary Impact Metrics (cl100k_base)")
    metric1, metric2, metric3, metric4 = st.columns(4)
    
    bantu_tax_pct = ((shn_metrics["Total Tokens"] - eng_metrics["Total Tokens"]) / max(eng_metrics["Total Tokens"], 1)) * 100
    cost_diff = shn_metrics["Cost ($)"] - eng_metrics["Cost ($)"]
    avg_frag = shn_metrics["Avg Fragments/Word"]
    char_density_diff = eng_metrics["Char/Token Density"] - shn_metrics["Char/Token Density"]
    
    metric1.metric("Total Bantu Tax (%)", f"{bantu_tax_pct:+.2f}%")
    metric2.metric("Dollar Cost Difference", f"${cost_diff:+.6f}")
    metric3.metric("Shona Avg Fragments", f"{avg_frag:.2f}")
    metric4.metric("Density Deficit (Chars/Token)", f"-{char_density_diff:.2f}")
    
    # Section 3: Advanced Visual Analytics
    st.header("3. Advanced Efficiency Analytics")
    
    tab1, tab2, tab3 = st.tabs(["Context Window Footprint", "Interactive Cost Scale Sandbox", "Density & Glitches"])
    
    with tab1:
        st.subheader("LLM Context Window Consumption")
        st.markdown(f"If we put this text into a 128k context window, notice how much faster Shona exhausts the LLM's memory compared to English.")
        
        ctx_size = 128_000
        eng_pct = (eng_metrics["Total Tokens"] / ctx_size) * 100
        shn_pct = (shn_metrics["Total Tokens"] / ctx_size) * 100
        
        chart_data = pd.DataFrame(
            {
                "Language": ["English", "Shona"],
                "% of 128k Context": [eng_pct, shn_pct],
            }
        ).set_index("Language")
        st.bar_chart(chart_data)
        
    with tab2:
        st.subheader("Enterprise Cost Sandbox")
        words_slider = st.select_slider(
            "Predict cost for processing a target word count:",
            options=[10_000, 100_000, 500_000, 1_000_000, 10_000_000, 50_000_000]
        )
        
        # Scale ratios
        eng_cost_per_word = eng_metrics["Cost ($)"] / eng_metrics["Total Words"] if eng_metrics["Total Words"] > 0 else 0
        shona_cost_per_word = shn_metrics["Cost ($)"] / shn_metrics["Total Words"] if shn_metrics["Total Words"] > 0 else 0
        
        proj_eng_cost = eng_cost_per_word * words_slider
        proj_shn_cost = shona_cost_per_word * words_slider
        
        st.info(f"At **{words_slider:,} words**, English costs **${proj_eng_cost:,.2f}** while Shona costs **${proj_shn_cost:,.2f}**.")
        st.markdown(f"**Enterprise Penalty:** processing Shona costs **${proj_shn_cost - proj_eng_cost:,.2f}** more for the exact same semantic volume.")

    with tab3:
        st.subheader("Token-to-Word Analysis")
        colA, colB = st.columns(2)
        with colA:
            st.metric("English Fertility (Tokens/Word)", f"{eng_metrics['Fertility Ratio']:.3f} T/W")
            st.metric("English Density", f"{eng_metrics['Char/Token Density']:.2f} Chars/Token")
            st.metric("English Glitches (>4 splits)", f"{eng_metrics['Deep Fragmentation Glitches']}")
        with colB:
            st.metric("Shona Fertility (Tokens/Word)", f"{shn_metrics['Fertility Ratio']:.3f} T/W")
            st.metric("Shona Density", f"{shn_metrics['Char/Token Density']:.2f} Chars/Token")
            st.metric("Shona Glitches (>4 splits)", f"{shn_metrics['Deep Fragmentation Glitches']}")


    # Section 4: Worst Offenders Table
    st.header("4. Top 10 Most Fragmented Words")
    st.markdown("Words that were shattered into >4 meaningless sub-tokens, destroying semantic root tracking.")
    
    if shn_metrics["Worst Offenders"]:
        df_worst = pd.DataFrame(shn_metrics["Worst Offenders"])
        st.dataframe(df_worst, use_container_width=True)
    else:
        st.success("No severely fragmented words found in the uploaded text!")

    # Section 5: Raw Analysis DataFrame
    st.header("5. Raw Analysis Comparison")
    
    df = pd.DataFrame([
        {k: v for k, v in analyze_text(eng_text, "English (cl100k)", enc_cl100k).items() if k != "Worst Offenders"},
        {k: v for k, v in analyze_text(shn_text, "Shona (cl100k)", enc_cl100k).items() if k != "Worst Offenders"},
        {k: v for k, v in analyze_text(eng_text, "English (o200k)", enc_o200k).items() if k != "Worst Offenders"},
        {k: v for k, v in analyze_text(shn_text, "Shona (o200k)", enc_o200k).items() if k != "Worst Offenders"},
    ])
    
    df_display = df.copy()
    df_display["Cost ($)"] = df_display["Cost ($)"].apply(lambda x: f"${x:.6f}")
    st.dataframe(df_display, use_container_width=True)
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Raw Report (CSV)", data=csv, file_name='tokenization_report.csv', mime='text/csv')

elif eng_file or shn_file:
    st.info("Please upload both English and Shona text files to see the comparison.")

st.header("6. Forensic Sandbox (Deep Fragmentation)")
st.markdown("Type any Shona word to visualize how the model mangles it.")

test_word = st.text_input("Enter Word:")
if test_word:
    col_fc, col_fo = st.columns(2)
    colors = ["#ff7b72", "#79c0ff", "#d2a8ff", "#a5d6ff", "#ffa657"]
    
    with col_fc:
        st.subheader("cl100k_base (GPT-4)")
        tokens_cl100k = enc_cl100k.encode(test_word)
        frags_cl100k = [enc_cl100k.decode([t]) for t in tokens_cl100k]
        
        html_cl100k = ""
        for i, frag in enumerate(frags_cl100k):
            col = colors[i % len(colors)]
            html_cl100k += f'<span style="background-color: {col}; color: #0d1117; padding: 2px 5px; margin: 2px; border-radius: 3px; font-weight: bold;">{frag}</span>'
        
        st.markdown(f"**Tokens:** {len(tokens_cl100k)} <br><br> **Sub-words:** <br>{html_cl100k}", unsafe_allow_html=True)
        
    with col_fo:
        st.subheader("o200k_base (GPT-4o)")
        tokens_o200k = enc_o200k.encode(test_word)
        frags_o200k = [enc_o200k.decode([t]) for t in tokens_o200k]
        
        html_o200k = ""
        for i, frag in enumerate(frags_o200k):
            col = colors[i % len(colors)]
            html_o200k += f'<span style="background-color: {col}; color: #0d1117; padding: 2px 5px; margin: 2px; border-radius: 3px; font-weight: bold;">{frag}</span>'
        
        st.markdown(f"**Tokens:** {len(tokens_o200k)} <br><br> **Sub-words:** <br>{html_o200k}", unsafe_allow_html=True)
