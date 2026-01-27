# Claim Analyzer

Extract factual claims from text and label their uncertainty levels based on language analysis.

## What It Does

- Extracts factual assertions from any text
- Labels uncertainty: `certain`, `likely`, `uncertain`, `speculative`
- Returns structured JSON output

## Quick Start

**Web Interface:**
```bash
streamlit run app.py
```

**Command Line Tests:**
```bash
python test_analyzer.py        # Normal cases
python break_analyzer.py       # Edge cases
```

## Key Design Decisions

### 1. Extracts all claims, even contradictory ones
If text says "sky is blue" AND "sky is red", both are extracted. The job is to identify claims made in text, not judge their truth. Useful for detecting contradictions or analyzing unreliable sources.

### 2. Multi-stage pipeline (validate → extract → post-process)
Separates input validation, LLM extraction, and output validation. Makes debugging systematic - you know exactly which stage failed. Each component is testable independently.

### 3. Intelligent retry logic
- **Retries**: Rate limits, server errors (transient failures)
- **Fails immediately**: Bad requests, JSON errors (won't fix with retry)

Optimizes for both reliability and cost.

### 4. Uncertainty is language-based, not truth-based
"certain" means definitive wording, NOT factual accuracy. "The Earth is flat" gets labeled "certain" if stated definitively.

## Known Limitations

- No fact-checking or verification
- Vague claims lose meaning without context ("It happened")
- No contradiction resolution
- Cost scales with text length (~1,500 tokens per 10K chars)
- Plain text only (no PDFs, images, rich media)

## Project Structure

```
claim-analyzer/
├── config.py           # Settings & constants
├── analyzer.py         # Core pipeline logic
├── test_analyzer.py    # Standard tests
├── break_analyzer.py   # Adversarial tests
└── app.py             # Streamlit web UI
```

## Future Improvements

- Add claim similarity detection (group duplicate/similar claims)
- Support batch processing for multiple documents
- Add confidence scores based on linguistic features
- Implement caching for repeated analyses
- Add export formats (CSV, PDF reports)