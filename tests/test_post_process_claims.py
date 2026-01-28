"""Unit tests for post_process_claims() function."""

import pytest
from analyzer import post_process_claims
from config import VALID_UNCERTAINTY_LABELS


class TestPostProcessClaimsEmptyInput:
    """Tests for empty/invalid input handling."""

    def test_empty_list_returns_empty(self):
        """Empty claims list should return empty list."""
        result = post_process_claims([])
        assert result == []

    def test_none_items_are_skipped(self):
        """None items in claims list should be skipped."""
        claims = [None, {"claim": "Test claim", "uncertainty": "certain"}]
        result = post_process_claims(claims)
        assert len(result) == 1

    def test_non_dict_items_are_skipped(self):
        """Non-dict items should be skipped."""
        claims = [
            "string claim",
            123,
            ["list", "claim"],
            {"claim": "Valid claim", "uncertainty": "certain"}
        ]
        result = post_process_claims(claims)
        assert len(result) == 1
        assert result[0]["claim"] == "Valid claim"


class TestPostProcessClaimsMissingFields:
    """Tests for claims with missing required fields."""

    def test_missing_claim_field_is_skipped(self):
        """Claims without 'claim' field should be skipped."""
        claims = [{"uncertainty": "certain", "context": "Some context"}]
        result = post_process_claims(claims)
        assert result == []

    def test_missing_uncertainty_field_is_skipped(self):
        """Claims without 'uncertainty' field should be skipped."""
        claims = [{"claim": "Test claim", "context": "Some context"}]
        result = post_process_claims(claims)
        assert result == []

    def test_both_fields_missing_is_skipped(self):
        """Claims without both required fields should be skipped."""
        claims = [{"context": "Only context provided"}]
        result = post_process_claims(claims)
        assert result == []

    def test_empty_claim_value_is_kept(self):
        """Empty string claim value should still be processed."""
        claims = [{"claim": "", "uncertainty": "certain"}]
        result = post_process_claims(claims)
        assert len(result) == 1
        assert result[0]["claim"] == ""


class TestPostProcessClaimsUncertaintyLabels:
    """Tests for uncertainty label validation and normalization."""

    def test_valid_certain_label(self):
        """'certain' label should be preserved."""
        claims = [{"claim": "Test", "uncertainty": "certain"}]
        result = post_process_claims(claims)
        assert result[0]["uncertainty"] == "certain"

    def test_valid_likely_label(self):
        """'likely' label should be preserved."""
        claims = [{"claim": "Test", "uncertainty": "likely"}]
        result = post_process_claims(claims)
        assert result[0]["uncertainty"] == "likely"

    def test_valid_uncertain_label(self):
        """'uncertain' label should be preserved."""
        claims = [{"claim": "Test", "uncertainty": "uncertain"}]
        result = post_process_claims(claims)
        assert result[0]["uncertainty"] == "uncertain"

    def test_valid_speculative_label(self):
        """'speculative' label should be preserved."""
        claims = [{"claim": "Test", "uncertainty": "speculative"}]
        result = post_process_claims(claims)
        assert result[0]["uncertainty"] == "speculative"

    def test_uppercase_label_normalized_to_lowercase(self):
        """Uppercase labels should be normalized to lowercase."""
        claims = [{"claim": "Test", "uncertainty": "CERTAIN"}]
        result = post_process_claims(claims)
        assert result[0]["uncertainty"] == "certain"

    def test_mixed_case_label_normalized(self):
        """Mixed case labels should be normalized."""
        claims = [{"claim": "Test", "uncertainty": "LiKeLy"}]
        result = post_process_claims(claims)
        assert result[0]["uncertainty"] == "likely"

    def test_invalid_label_defaults_to_uncertain(self):
        """Invalid uncertainty labels should default to 'uncertain'."""
        claims = [{"claim": "Test", "uncertainty": "definitely"}]
        result = post_process_claims(claims)
        assert result[0]["uncertainty"] == "uncertain"

    def test_empty_label_defaults_to_uncertain(self):
        """Empty string label should default to 'uncertain'."""
        claims = [{"claim": "Test", "uncertainty": ""}]
        result = post_process_claims(claims)
        assert result[0]["uncertainty"] == "uncertain"

    def test_numeric_label_defaults_to_uncertain(self):
        """Numeric label (if somehow passed) should default to 'uncertain'."""
        claims = [{"claim": "Test", "uncertainty": 123}]
        result = post_process_claims(claims)
        # Note: this will call .lower() on 123 which will fail
        # The code has get("uncertainty", "").lower() which will error on non-string
        # Let me check the actual behavior
        assert result[0]["uncertainty"] == "uncertain"


class TestPostProcessClaimsContextHandling:
    """Tests for context field handling."""

    def test_missing_context_defaults_to_empty_string(self):
        """Missing context should default to empty string."""
        claims = [{"claim": "Test", "uncertainty": "certain"}]
        result = post_process_claims(claims)
        assert result[0]["context"] == ""

    def test_context_is_preserved(self):
        """Provided context should be preserved."""
        claims = [{"claim": "Test", "uncertainty": "certain", "context": "Test context"}]
        result = post_process_claims(claims)
        assert result[0]["context"] == "Test context"

    def test_empty_context_is_preserved(self):
        """Explicitly empty context should be preserved."""
        claims = [{"claim": "Test", "uncertainty": "certain", "context": ""}]
        result = post_process_claims(claims)
        assert result[0]["context"] == ""


class TestPostProcessClaimsOutputStructure:
    """Tests for output structure validation."""

    def test_output_has_correct_keys(self):
        """Each output claim should have exactly claim, uncertainty, context."""
        claims = [{"claim": "Test", "uncertainty": "certain", "context": "Ctx", "extra": "ignored"}]
        result = post_process_claims(claims)
        assert set(result[0].keys()) == {"claim", "uncertainty", "context"}

    def test_extra_fields_are_removed(self):
        """Extra fields in input claims should not appear in output."""
        claims = [{
            "claim": "Test",
            "uncertainty": "certain",
            "context": "Ctx",
            "source": "ignored",
            "confidence": 0.95
        }]
        result = post_process_claims(claims)
        assert "source" not in result[0]
        assert "confidence" not in result[0]

    def test_multiple_claims_processed_in_order(self):
        """Multiple claims should be processed and returned in order."""
        claims = [
            {"claim": "First", "uncertainty": "certain"},
            {"claim": "Second", "uncertainty": "likely"},
            {"claim": "Third", "uncertainty": "speculative"}
        ]
        result = post_process_claims(claims)
        assert len(result) == 3
        assert result[0]["claim"] == "First"
        assert result[1]["claim"] == "Second"
        assert result[2]["claim"] == "Third"

    def test_mixed_valid_invalid_claims(self):
        """Should process valid claims and skip invalid ones."""
        claims = [
            {"claim": "Valid 1", "uncertainty": "certain"},
            {"missing": "claim field"},
            {"claim": "Valid 2", "uncertainty": "likely"},
            "not a dict",
            {"claim": "Valid 3", "uncertainty": "INVALID_LABEL"}
        ]
        result = post_process_claims(claims)
        assert len(result) == 3
        assert result[0]["claim"] == "Valid 1"
        assert result[1]["claim"] == "Valid 2"
        assert result[2]["claim"] == "Valid 3"
        assert result[2]["uncertainty"] == "uncertain"  # Invalid label defaulted


class TestPostProcessClaimsAllValidLabels:
    """Ensure all valid labels from config are accepted."""

    @pytest.mark.parametrize("label", VALID_UNCERTAINTY_LABELS)
    def test_all_config_labels_are_valid(self, label):
        """All labels defined in config should be accepted."""
        claims = [{"claim": "Test", "uncertainty": label}]
        result = post_process_claims(claims)
        assert result[0]["uncertainty"] == label
