"""Unit tests for validate_input() function."""

import pytest
from analyzer import validate_input
from config import MIN_TEXT_LENGTH, MAX_TEXT_LENGTH, DEFAULT_MAX_CLAIMS


class TestValidateInputTextType:
    """Tests for text type validation."""

    def test_non_string_input_raises_error(self):
        """Non-string inputs should raise ValueError."""
        with pytest.raises(ValueError, match="Input text must be a string"):
            validate_input(123)

    def test_none_input_raises_error(self):
        """None input should raise ValueError."""
        with pytest.raises(ValueError, match="Input text must be a string"):
            validate_input(None)

    def test_list_input_raises_error(self):
        """List input should raise ValueError."""
        with pytest.raises(ValueError, match="Input text must be a string"):
            validate_input(["some", "text"])

    def test_dict_input_raises_error(self):
        """Dict input should raise ValueError."""
        with pytest.raises(ValueError, match="Input text must be a string"):
            validate_input({"text": "value"})


class TestValidateInputEmptyText:
    """Tests for empty/whitespace text validation."""

    def test_empty_string_raises_error(self):
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError, match="Input text cannot be empty or whitespace only"):
            validate_input("")

    def test_whitespace_only_raises_error(self):
        """Whitespace-only string should raise ValueError."""
        with pytest.raises(ValueError, match="Input text cannot be empty or whitespace only"):
            validate_input("   ")

    def test_tabs_only_raises_error(self):
        """Tab-only string should raise ValueError."""
        with pytest.raises(ValueError, match="Input text cannot be empty or whitespace only"):
            validate_input("\t\t\t")

    def test_newlines_only_raises_error(self):
        """Newline-only string should raise ValueError."""
        with pytest.raises(ValueError, match="Input text cannot be empty or whitespace only"):
            validate_input("\n\n\n")

    def test_mixed_whitespace_only_raises_error(self):
        """Mixed whitespace string should raise ValueError."""
        with pytest.raises(ValueError, match="Input text cannot be empty or whitespace only"):
            validate_input("  \t\n  ")


class TestValidateInputTextLength:
    """Tests for text length validation."""

    def test_below_min_length_raises_error(self):
        """Text shorter than MIN_TEXT_LENGTH should raise ValueError."""
        short_text = "a" * (MIN_TEXT_LENGTH - 1)
        with pytest.raises(ValueError, match=f"Text too short for analysis \\(min {MIN_TEXT_LENGTH} chars\\)"):
            validate_input(short_text)

    def test_at_exact_min_length_succeeds(self):
        """Text exactly at MIN_TEXT_LENGTH should succeed."""
        min_text = "a" * MIN_TEXT_LENGTH
        result = validate_input(min_text)
        assert result["text"] == min_text
        assert result["text_length"] == MIN_TEXT_LENGTH

    def test_above_max_length_raises_error(self):
        """Text longer than MAX_TEXT_LENGTH should raise ValueError."""
        long_text = "a" * (MAX_TEXT_LENGTH + 1)
        with pytest.raises(ValueError, match=f"Text exceeds maximum length \\({MAX_TEXT_LENGTH} chars\\)"):
            validate_input(long_text)

    def test_at_exact_max_length_succeeds(self):
        """Text exactly at MAX_TEXT_LENGTH should succeed."""
        max_text = "a" * MAX_TEXT_LENGTH
        result = validate_input(max_text)
        assert result["text"] == max_text
        assert result["text_length"] == MAX_TEXT_LENGTH

    def test_normal_length_text_succeeds(self):
        """Text within valid range should succeed."""
        normal_text = "This is a normal length text for testing purposes."
        result = validate_input(normal_text)
        assert result["text"] == normal_text
        assert result["text_length"] == len(normal_text)


class TestValidateInputMaxClaims:
    """Tests for max_claims parameter validation."""

    def test_zero_max_claims_raises_error(self):
        """max_claims of 0 should raise ValueError."""
        with pytest.raises(ValueError, match="max_claims must be a positive integer"):
            validate_input("Valid text for testing", max_claims=0)

    def test_negative_max_claims_raises_error(self):
        """Negative max_claims should raise ValueError."""
        with pytest.raises(ValueError, match="max_claims must be a positive integer"):
            validate_input("Valid text for testing", max_claims=-5)

    def test_float_max_claims_raises_error(self):
        """Float max_claims should raise ValueError."""
        with pytest.raises(ValueError, match="max_claims must be a positive integer"):
            validate_input("Valid text for testing", max_claims=5.5)

    def test_string_max_claims_raises_error(self):
        """String max_claims should raise ValueError."""
        with pytest.raises(ValueError, match="max_claims must be a positive integer"):
            validate_input("Valid text for testing", max_claims="10")

    def test_valid_max_claims_succeeds(self):
        """Valid positive integer max_claims should succeed."""
        result = validate_input("Valid text for testing", max_claims=5)
        assert result["max_claims"] == 5

    def test_default_max_claims_used(self):
        """Default max_claims should be used when not specified."""
        result = validate_input("Valid text for testing")
        assert result["max_claims"] == DEFAULT_MAX_CLAIMS


class TestValidateInputReturnValue:
    """Tests for validate_input return value structure."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        result = validate_input("Valid text for testing")
        assert isinstance(result, dict)

    def test_returns_stripped_text(self):
        """Should return stripped text."""
        result = validate_input("  Valid text for testing  ")
        assert result["text"] == "Valid text for testing"

    def test_returns_correct_keys(self):
        """Should return dict with correct keys."""
        result = validate_input("Valid text for testing", max_claims=5)
        assert "text" in result
        assert "max_claims" in result
        assert "text_length" in result

    def test_text_length_matches_original_not_stripped(self):
        """text_length should be length of original text, not stripped."""
        original_text = "  Valid text  "
        result = validate_input(original_text)
        assert result["text_length"] == len(original_text)
