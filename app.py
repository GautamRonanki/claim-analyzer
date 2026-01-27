import streamlit as st
from analyzer import analyze_text
import json

st.set_page_config(
    page_title="Claim Analyzer",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Claim Analyzer")
st.markdown("""
Extract factual claims from any text and analyze their uncertainty levels.
This tool identifies assertions made in text without verifying their truth.
""")


st.sidebar.header("Settings")
max_claims = st.sidebar.slider(
    "Maximum claims to extract",
    min_value=1,
    max_value=20,
    value=10,
    help="Limit the number of claims extracted from the text"
)


st.sidebar.header("Load Example")

examples = {
    "None": "",
    "News Article": """The Federal Reserve raised interest rates by 0.25% yesterday. This move is expected to curb inflation, which has been running at 6.5% annually. Economists suggest the rate hike might slow economic growth in the coming months.""",
    
    "Product Review": """This laptop has a 15-inch display and weighs 3.2 pounds. The battery life appears to last around 8 hours under normal use. Performance seems excellent for most tasks, though it might struggle with heavy video editing.""",
    
    "Research Abstract": """The study found a correlation between sleep duration and cognitive performance. Participants who slept 7-8 hours scored 15% higher on memory tests. The researchers suggest that sleep quality may be more important than duration."""
}

selected_example = st.sidebar.selectbox(
    "Choose an example:",
    options=list(examples.keys()),
    index=0
)


st.header("Input Text")

input_text = st.text_area(
    "Paste your text here:",
    value=examples[selected_example], 
    height=200,
    placeholder="Enter text to analyze (10-10,000 characters)...",
    help="Supports news articles, reviews, research abstracts, or any text content"
)


char_count = len(input_text)
st.caption(f"Characters: {char_count}/10,000")


if st.button("🔍 Analyze Claims", type="primary"):
    if not input_text.strip():
        st.error("⚠️ Please enter some text to analyze")
    else:
        with st.spinner("Analyzing text..."):
            result = analyze_text(input_text, max_claims=max_claims)
        
 
        if "error" in result:
            st.error(f"❌ Error: {result['error']}")
            st.caption(f"Error type: {result['error_type']}")
        else:
 
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Claims Found", result['metadata']['total_claims_found'])
            with col2:
                st.metric("Tokens Used", result['metadata']['tokens_used'])
            with col3:
                st.metric("Text Length", result['metadata']['text_length'])
            

            st.header("📋 Extracted Claims")
            
            if result['claims']:

                uncertainty_colors = {
                    'certain': '🟢',
                    'likely': '🟡',
                    'uncertain': '🟠',
                    'speculative': '🔴'
                }
                
                for i, claim in enumerate(result['claims'], 1):
                    uncertainty = claim['uncertainty']
                    icon = uncertainty_colors.get(uncertainty, '⚪')
                    
                    with st.expander(f"{icon} Claim {i}: {claim['claim'][:60]}...", expanded=(i <= 3)):
                        st.markdown(f"**Claim:** {claim['claim']}")
                        st.markdown(f"**Uncertainty Level:** `{uncertainty.upper()}`")
                        if claim['context']:
                            st.markdown(f"**Context:** _{claim['context']}_")
            else:
                st.info("ℹ️ No claims found in the provided text")
            

            with st.expander("📄 View Raw JSON Output"):
                st.json(result)


st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
This analyzer extracts factual claims from text and labels their uncertainty based on language used.

**Uncertainty Levels:**
- 🟢 Certain: Definitive statements
- 🟡 Likely: Probable statements
- 🟠 Uncertain: Tentative statements
- 🔴 Speculative: Highly uncertain
""")