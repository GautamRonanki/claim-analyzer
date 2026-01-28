"""Pytest configuration and shared fixtures."""

import pytest
import os

# Set the API key before any modules are imported
# This must happen at module level, before pytest collects tests
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-testing")


@pytest.fixture
def sample_claims():
    """Sample claims data for testing."""
    return [
        {"claim": "The sky is blue", "uncertainty": "certain", "context": "Weather observation"},
        {"claim": "It might rain tomorrow", "uncertainty": "uncertain", "context": "Weather forecast"},
        {"claim": "Studies suggest caffeine improves focus", "uncertainty": "likely", "context": "Research findings"},
    ]


@pytest.fixture
def sample_text():
    """Sample text for analysis testing."""
    return """
    The Federal Reserve raised interest rates by 0.25% yesterday.
    This move is expected to curb inflation, which has been running at 6.5% annually.
    Economists suggest the rate hike might slow economic growth in the coming months.
    """


@pytest.fixture
def min_length_text():
    """Text at minimum valid length."""
    from config import MIN_TEXT_LENGTH
    return "a" * MIN_TEXT_LENGTH


@pytest.fixture
def max_length_text():
    """Text at maximum valid length."""
    from config import MAX_TEXT_LENGTH
    return "a" * MAX_TEXT_LENGTH
