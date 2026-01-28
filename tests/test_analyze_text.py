"""Integration tests for analyze_text() with assertions."""

import pytest
from unittest.mock import Mock, patch
import json
from analyzer import analyze_text
from config import MIN_TEXT_LENGTH, MAX_TEXT_LENGTH, DEFAULT_MAX_CLAIMS


def create_mock_llm_response(claims):
    """Helper to create mock LLM response."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = json.dumps({"claims": claims})
    mock_response.usage = Mock()
    mock_response.usage.total_tokens = 150
    return mock_response


class TestAnalyzeTextValidationErrors:
    """Tests for input validation error handling."""

    def test_empty_string_returns_validation_error(self):
        """Empty string should return validation error."""
        result = analyze_text("")

        assert "error" in result
        assert result["error_type"] == "validation_error"
        assert "empty" in result["error"].lower()

    def test_too_short_text_returns_validation_error(self):
        """Text below minimum length should return validation error."""
        short_text = "a" * (MIN_TEXT_LENGTH - 1)
        result = analyze_text(short_text)

        assert "error" in result
        assert result["error_type"] == "validation_error"
        assert "too short" in result["error"].lower()

    def test_too_long_text_returns_validation_error(self):
        """Text above maximum length should return validation error."""
        long_text = "a" * (MAX_TEXT_LENGTH + 1)
        result = analyze_text(long_text)

        assert "error" in result
        assert result["error_type"] == "validation_error"
        assert "exceeds" in result["error"].lower()

    def test_invalid_max_claims_returns_validation_error(self):
        """Invalid max_claims should return validation error."""
        result = analyze_text("Valid text for testing", max_claims=0)

        assert "error" in result
        assert result["error_type"] == "validation_error"

    def test_whitespace_only_returns_validation_error(self):
        """Whitespace-only text should return validation error."""
        result = analyze_text("   \t\n   ")

        assert "error" in result
        assert result["error_type"] == "validation_error"


class TestAnalyzeTextSuccessfulAnalysis:
    """Tests for successful text analysis."""

    @patch('analyzer.client')
    def test_successful_analysis_returns_claims(self, mock_client):
        """Successful analysis should return claims structure."""
        claims = [
            {"claim": "The sky is blue", "uncertainty": "certain", "context": ""}
        ]
        mock_client.chat.completions.create.return_value = create_mock_llm_response(claims)

        result = analyze_text("The sky is blue and clear today.")

        assert "error" not in result
        assert "claims" in result
        assert len(result["claims"]) == 1

    @patch('analyzer.client')
    def test_successful_analysis_returns_metadata(self, mock_client):
        """Successful analysis should return metadata."""
        mock_client.chat.completions.create.return_value = create_mock_llm_response([])

        result = analyze_text("Some text for testing purposes here.")

        assert "metadata" in result
        assert "total_claims_found" in result["metadata"]
        assert "text_length" in result["metadata"]
        assert "analysis_timestamp" in result["metadata"]
        assert "tokens_used" in result["metadata"]
        assert "model" in result["metadata"]

    @patch('analyzer.client')
    def test_text_length_in_metadata(self, mock_client):
        """Metadata should contain correct text length."""
        mock_client.chat.completions.create.return_value = create_mock_llm_response([])

        test_text = "Test text for analysis."
        result = analyze_text(test_text)

        assert result["metadata"]["text_length"] == len(test_text)

    @patch('analyzer.client')
    def test_claims_count_in_metadata(self, mock_client):
        """Metadata should contain correct claims count."""
        claims = [
            {"claim": "Claim 1", "uncertainty": "certain", "context": ""},
            {"claim": "Claim 2", "uncertainty": "likely", "context": ""}
        ]
        mock_client.chat.completions.create.return_value = create_mock_llm_response(claims)

        result = analyze_text("Text with multiple claims to extract.")

        assert result["metadata"]["total_claims_found"] == 2


class TestAnalyzeTextPostProcessing:
    """Tests for post-processing of claims."""

    @patch('analyzer.client')
    def test_invalid_uncertainty_label_normalized(self, mock_client):
        """Invalid uncertainty labels should be normalized to 'uncertain'."""
        claims = [
            {"claim": "Test claim", "uncertainty": "INVALID_LABEL", "context": ""}
        ]
        mock_client.chat.completions.create.return_value = create_mock_llm_response(claims)

        result = analyze_text("Some text for testing claims.")

        assert result["claims"][0]["uncertainty"] == "uncertain"

    @patch('analyzer.client')
    def test_uppercase_uncertainty_normalized(self, mock_client):
        """Uppercase uncertainty labels should be normalized."""
        claims = [
            {"claim": "Test claim", "uncertainty": "CERTAIN", "context": ""}
        ]
        mock_client.chat.completions.create.return_value = create_mock_llm_response(claims)

        result = analyze_text("Some text for testing claims.")

        assert result["claims"][0]["uncertainty"] == "certain"

    @patch('analyzer.client')
    def test_malformed_claims_filtered_out(self, mock_client):
        """Malformed claims should be filtered out."""
        claims = [
            {"claim": "Valid claim", "uncertainty": "certain", "context": ""},
            {"missing_claim_key": "value", "uncertainty": "certain"},  # Invalid
            {"claim": "Another valid", "uncertainty": "likely", "context": ""}
        ]
        mock_client.chat.completions.create.return_value = create_mock_llm_response(claims)

        result = analyze_text("Text with various claim structures.")

        assert len(result["claims"]) == 2
        assert result["claims"][0]["claim"] == "Valid claim"
        assert result["claims"][1]["claim"] == "Another valid"


class TestAnalyzeTextAPIErrors:
    """Tests for API error handling."""

    @patch('analyzer.client')
    def test_api_error_returns_analysis_error(self, mock_client):
        """API errors should return analysis_error type."""
        mock_client.chat.completions.create.side_effect = Exception("API Failed")

        result = analyze_text("Valid text for testing purposes.")

        assert "error" in result
        assert result["error_type"] == "analysis_error"

    @patch('analyzer.client')
    def test_json_parse_error_returns_error(self, mock_client):
        """JSON parsing errors should return an error."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "not valid json"
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 100
        mock_client.chat.completions.create.return_value = mock_response

        result = analyze_text("Valid text for testing purposes.")

        assert "error" in result
        # JSON errors are wrapped as ValueError in extract_claims, caught as validation_error
        assert result["error_type"] in ["validation_error", "analysis_error"]
        assert "JSON" in result["error"] or "json" in result["error"].lower()


class TestAnalyzeTextMaxClaimsParameter:
    """Tests for max_claims parameter."""

    @patch('analyzer.client')
    def test_default_max_claims_used(self, mock_client):
        """Default max_claims should be used when not specified."""
        mock_client.chat.completions.create.return_value = create_mock_llm_response([])

        analyze_text("Valid text for testing purposes.")

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs['messages']
        user_content = messages[1]['content']
        assert str(DEFAULT_MAX_CLAIMS) in user_content

    @patch('analyzer.client')
    def test_custom_max_claims_passed(self, mock_client):
        """Custom max_claims should be passed to extraction."""
        mock_client.chat.completions.create.return_value = create_mock_llm_response([])

        analyze_text("Valid text for testing purposes.", max_claims=15)

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs['messages']
        user_content = messages[1]['content']
        assert "15" in user_content


class TestAnalyzeTextEdgeCases:
    """Tests for edge cases."""

    @patch('analyzer.client')
    def test_minimum_length_text_succeeds(self, mock_client):
        """Text at minimum length should succeed."""
        mock_client.chat.completions.create.return_value = create_mock_llm_response([])

        min_text = "a" * MIN_TEXT_LENGTH
        result = analyze_text(min_text)

        assert "error" not in result

    @patch('analyzer.client')
    def test_maximum_length_text_succeeds(self, mock_client):
        """Text at maximum length should succeed."""
        mock_client.chat.completions.create.return_value = create_mock_llm_response([])

        max_text = "a" * MAX_TEXT_LENGTH
        result = analyze_text(max_text)

        assert "error" not in result

    @patch('analyzer.client')
    def test_empty_claims_result(self, mock_client):
        """Text with no claims should return empty claims array."""
        mock_client.chat.completions.create.return_value = create_mock_llm_response([])

        result = analyze_text("Just a question? Maybe nothing here...")

        assert "error" not in result
        assert result["claims"] == []
        assert result["metadata"]["total_claims_found"] == 0

    @patch('analyzer.client')
    def test_timestamp_is_iso_format(self, mock_client):
        """Timestamp should be in ISO format."""
        mock_client.chat.completions.create.return_value = create_mock_llm_response([])

        result = analyze_text("Test text for timestamp check.")

        timestamp = result["metadata"]["analysis_timestamp"]
        # ISO format check: contains T separator and proper structure
        assert "T" in timestamp
        assert len(timestamp) >= 19  # Minimum ISO format length


class TestAnalyzeTextClaimStructure:
    """Tests for claim structure in output."""

    @patch('analyzer.client')
    def test_claim_has_required_fields(self, mock_client):
        """Each claim should have claim, uncertainty, and context fields."""
        claims = [
            {"claim": "Test claim", "uncertainty": "certain", "context": "Test context"}
        ]
        mock_client.chat.completions.create.return_value = create_mock_llm_response(claims)

        result = analyze_text("Some text with a test claim.")

        claim = result["claims"][0]
        assert "claim" in claim
        assert "uncertainty" in claim
        assert "context" in claim

    @patch('analyzer.client')
    def test_all_uncertainty_levels_preserved(self, mock_client):
        """All valid uncertainty levels should be preserved in output."""
        claims = [
            {"claim": "C1", "uncertainty": "certain", "context": ""},
            {"claim": "C2", "uncertainty": "likely", "context": ""},
            {"claim": "C3", "uncertainty": "uncertain", "context": ""},
            {"claim": "C4", "uncertainty": "speculative", "context": ""}
        ]
        mock_client.chat.completions.create.return_value = create_mock_llm_response(claims)

        result = analyze_text("Text with multiple claim types.")

        uncertainties = [c["uncertainty"] for c in result["claims"]]
        assert set(uncertainties) == {"certain", "likely", "uncertain", "speculative"}
