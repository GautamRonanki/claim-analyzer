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

**Run Tests:**
```bash
# Run the full pytest test suite (105 tests)
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_validate_input.py -v
python -m pytest tests/test_analyze_text.py -v

# Run with coverage (requires pytest-cov)
python -m pytest tests/ --cov=. --cov-report=term-missing

# Legacy manual test scripts
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
├── app.py              # Streamlit web UI
├── pytest.ini          # Pytest configuration
├── test_analyzer.py    # Legacy manual tests
├── break_analyzer.py   # Legacy adversarial tests
└── tests/              # Pytest test suite
    ├── conftest.py               # Shared fixtures
    ├── test_validate_input.py    # Input validation tests (27)
    ├── test_post_process_claims.py # Post-processing tests (23)
    ├── test_call_llm_with_retry.py # Retry logic tests (17)
    ├── test_extract_claims.py    # Extraction tests (14)
    └── test_analyze_text.py      # Integration tests (22)
```

## Future Improvements

- Add claim similarity detection (group duplicate/similar claims)
- Support batch processing for multiple documents
- Add confidence scores based on linguistic features
- Implement caching for repeated analyses
- Add export formats (CSV, PDF reports)