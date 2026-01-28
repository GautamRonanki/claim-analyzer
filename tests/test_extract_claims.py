"""Unit tests for extract_claims() function with mocked LLM responses."""

import pytest
import json
from unittest.mock import Mock, patch
from analyzer import extract_claims
from config import MODEL


def create_mock_response(claims_data, total_tokens=150):
    """Create a mock OpenAI API response with claims data."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = json.dumps(claims_data)
    mock_response.usage = Mock()
    mock_response.usage.total_tokens = total_tokens
    return mock_response


class TestExtractClaimsValidResponses:
    """Tests for valid LLM responses."""

    @patch('analyzer.call_llm_with_retry')
    def test_extracts_single_claim(self, mock_llm):
        """Should extract a single claim correctly."""
        claims_data = {
            "claims": [{
                "claim": "The sky is blue",
                "uncertainty": "certain",
                "context": "Looking up at the sky"
            }]
        }
        mock_llm.return_value = create_mock_response(claims_data)

        result = extract_claims("The sky is blue.", max_claims=5)

        assert len(result["claims"]) == 1
        assert result["claims"][0]["claim"] == "The sky is blue"
        assert result["claims"][0]["uncertainty"] == "certain"

    @patch('analyzer.call_llm_with_retry')
    def test_extracts_multiple_claims(self, mock_llm):
        """Should extract multiple claims correctly."""
        claims_data = {
            "claims": [
                {"claim": "Claim 1", "uncertainty": "certain", "context": ""},
                {"claim": "Claim 2", "uncertainty": "likely", "context": ""},
                {"claim": "Claim 3", "uncertainty": "uncertain", "context": ""}
            ]
        }
        mock_llm.return_value = create_mock_response(claims_data)

        result = extract_claims("Some text with claims.", max_claims=5)

        assert len(result["claims"]) == 3

    @patch('analyzer.call_llm_with_retry')
    def test_empty_claims_array(self, mock_llm):
        """Should handle empty claims array."""
        claims_data = {"claims": []}
        mock_llm.return_value = create_mock_response(claims_data)

        result = extract_claims("Text with no claims?", max_claims=5)

        assert result["claims"] == []

    @patch('analyzer.call_llm_with_retry')
    def test_missing_claims_key_defaults_to_empty(self, mock_llm):
        """Should default to empty list if claims key is missing."""
        claims_data = {"other_key": "value"}
        mock_llm.return_value = create_mock_response(claims_data)

        result = extract_claims("Some text.", max_claims=5)

        assert result["claims"] == []


class TestExtractClaimsMetadata:
    """Tests for metadata in response."""

    @patch('analyzer.call_llm_with_retry')
    def test_returns_token_count(self, mock_llm):
        """Should return token usage count."""
        claims_data = {"claims": []}
        mock_llm.return_value = create_mock_response(claims_data, total_tokens=250)

        result = extract_claims("Test text.", max_claims=5)

        assert result["tokens_used"] == 250

    @patch('analyzer.call_llm_with_retry')
    def test_returns_model_name(self, mock_llm):
        """Should return model name from config."""
        claims_data = {"claims": []}
        mock_llm.return_value = create_mock_response(claims_data)

        result = extract_claims("Test text.", max_claims=5)

        assert result["model"] == MODEL


class TestExtractClaimsPromptConstruction:
    """Tests for prompt construction."""

    @patch('analyzer.call_llm_with_retry')
    def test_max_claims_included_in_prompt(self, mock_llm):
        """max_claims should be included in user prompt."""
        claims_data = {"claims": []}
        mock_llm.return_value = create_mock_response(claims_data)

        extract_claims("Test text.", max_claims=7)

        call_args = mock_llm.call_args
        messages = call_args.kwargs.get('messages') or call_args[0][0]
        user_message = messages[1]["content"]
        assert "7" in user_message

    @patch('analyzer.call_llm_with_retry')
    def test_text_included_in_prompt(self, mock_llm):
        """Input text should be included in user prompt."""
        claims_data = {"claims": []}
        mock_llm.return_value = create_mock_response(claims_data)

        test_text = "Unique test text for verification"
        extract_claims(test_text, max_claims=5)

        call_args = mock_llm.call_args
        messages = call_args.kwargs.get('messages') or call_args[0][0]
        user_message = messages[1]["content"]
        assert test_text in user_message

    @patch('analyzer.call_llm_with_retry')
    def test_system_prompt_contains_uncertainty_labels(self, mock_llm):
        """System prompt should contain uncertainty label definitions."""
        claims_data = {"claims": []}
        mock_llm.return_value = create_mock_response(claims_data)

        extract_claims("Test text.", max_claims=5)

        call_args = mock_llm.call_args
        messages = call_args.kwargs.get('messages') or call_args[0][0]
        system_message = messages[0]["content"]
        assert "certain" in system_message
        assert "likely" in system_message
        assert "uncertain" in system_message
        assert "speculative" in system_message


class TestExtractClaimsErrorHandling:
    """Tests for error handling."""

    @patch('analyzer.call_llm_with_retry')
    def test_invalid_json_raises_value_error(self, mock_llm):
        """Invalid JSON response should raise ValueError."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "not valid json"
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 100
        mock_llm.return_value = mock_response

        with pytest.raises(ValueError, match="LLM returned invalid JSON"):
            extract_claims("Test text.", max_claims=5)

    @patch('analyzer.call_llm_with_retry')
    def test_llm_exception_propagates(self, mock_llm):
        """Exceptions from LLM call should be wrapped."""
        mock_llm.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="LLM analysis failed"):
            extract_claims("Test text.", max_claims=5)

    @patch('analyzer.call_llm_with_retry')
    def test_malformed_json_structure(self, mock_llm):
        """Malformed but valid JSON should be handled gracefully."""
        # Valid JSON but unexpected structure
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"unexpected": "structure"}'
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 100
        mock_llm.return_value = mock_response

        result = extract_claims("Test text.", max_claims=5)

        # Should return empty claims due to .get("claims", [])
        assert result["claims"] == []


class TestExtractClaimsTemperature:
    """Tests for temperature parameter."""

    @patch('analyzer.call_llm_with_retry')
    def test_default_temperature_is_0_3(self, mock_llm):
        """Default temperature should be 0.3."""
        claims_data = {"claims": []}
        mock_llm.return_value = create_mock_response(claims_data)

        extract_claims("Test text.", max_claims=5)

        call_args = mock_llm.call_args
        assert call_args.kwargs.get('temperature') == 0.3


class TestExtractClaimsClaimStructure:
    """Tests for claim structure in response."""

    @patch('analyzer.call_llm_with_retry')
    def test_claim_with_all_fields(self, mock_llm):
        """Claims with all fields should be preserved."""
        claims_data = {
            "claims": [{
                "claim": "Test claim text",
                "uncertainty": "likely",
                "context": "Surrounding context"
            }]
        }
        mock_llm.return_value = create_mock_response(claims_data)

        result = extract_claims("Test text.", max_claims=5)

        assert result["claims"][0]["claim"] == "Test claim text"
        assert result["claims"][0]["uncertainty"] == "likely"
        assert result["claims"][0]["context"] == "Surrounding context"

    @patch('analyzer.call_llm_with_retry')
    def test_preserves_all_uncertainty_levels(self, mock_llm):
        """All uncertainty levels should be preserved."""
        claims_data = {
            "claims": [
                {"claim": "C1", "uncertainty": "certain", "context": ""},
                {"claim": "C2", "uncertainty": "likely", "context": ""},
                {"claim": "C3", "uncertainty": "uncertain", "context": ""},
                {"claim": "C4", "uncertainty": "speculative", "context": ""}
            ]
        }
        mock_llm.return_value = create_mock_response(claims_data)

        result = extract_claims("Test text.", max_claims=10)

        uncertainties = [c["uncertainty"] for c in result["claims"]]
        assert "certain" in uncertainties
        assert "likely" in uncertainties
        assert "uncertain" in uncertainties
        assert "speculative" in uncertainties
