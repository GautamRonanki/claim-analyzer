"""Unit tests for call_llm_with_retry() function with mocked OpenAI client."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from openai import (
    RateLimitError,
    APIConnectionError,
    InternalServerError,
    BadRequestError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError
)
from analyzer import call_llm_with_retry


def create_mock_response(content='{"claims": []}', total_tokens=100):
    """Create a mock OpenAI API response."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = content
    mock_response.usage = Mock()
    mock_response.usage.total_tokens = total_tokens
    return mock_response


def create_mock_http_response(status_code):
    """Create a mock httpx Response object."""
    mock_response = Mock()
    mock_response.status_code = status_code
    mock_response.headers = {}
    mock_response.text = "Error"
    mock_response.request = Mock()
    return mock_response


def create_api_error(error_class, message="Test error"):
    """Create mock API errors with required attributes."""
    if error_class == RateLimitError:
        return RateLimitError(
            message,
            response=create_mock_http_response(429),
            body={"error": {"message": message}}
        )
    elif error_class == APIConnectionError:
        return APIConnectionError(request=Mock())
    elif error_class == InternalServerError:
        return InternalServerError(
            message,
            response=create_mock_http_response(500),
            body={"error": {"message": message}}
        )
    elif error_class == BadRequestError:
        return BadRequestError(
            message,
            response=create_mock_http_response(400),
            body={"error": {"message": message}}
        )
    elif error_class == AuthenticationError:
        return AuthenticationError(
            message,
            response=create_mock_http_response(401),
            body={"error": {"message": message}}
        )
    elif error_class == PermissionDeniedError:
        return PermissionDeniedError(
            message,
            response=create_mock_http_response(403),
            body={"error": {"message": message}}
        )
    elif error_class == NotFoundError:
        return NotFoundError(
            message,
            response=create_mock_http_response(404),
            body={"error": {"message": message}}
        )


class TestCallLLMSuccess:
    """Tests for successful API calls."""

    @patch('analyzer.client')
    def test_successful_call_returns_response(self, mock_client):
        """Successful API call should return response."""
        mock_response = create_mock_response()
        mock_client.chat.completions.create.return_value = mock_response

        messages = [{"role": "user", "content": "Test"}]
        result = call_llm_with_retry(messages)

        assert result == mock_response
        mock_client.chat.completions.create.assert_called_once()

    @patch('analyzer.client')
    def test_temperature_parameter_passed(self, mock_client):
        """Temperature parameter should be passed to API."""
        mock_client.chat.completions.create.return_value = create_mock_response()

        messages = [{"role": "user", "content": "Test"}]
        call_llm_with_retry(messages, temperature=0.7)

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs['temperature'] == 0.7

    @patch('analyzer.client')
    def test_json_response_format_requested(self, mock_client):
        """JSON response format should be requested."""
        mock_client.chat.completions.create.return_value = create_mock_response()

        messages = [{"role": "user", "content": "Test"}]
        call_llm_with_retry(messages)

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs['response_format'] == {"type": "json_object"}


class TestCallLLMRetryableErrors:
    """Tests for retryable error handling."""

    @patch('analyzer.time.sleep')
    @patch('analyzer.client')
    def test_rate_limit_error_triggers_retry(self, mock_client, mock_sleep):
        """RateLimitError should trigger retry."""
        mock_client.chat.completions.create.side_effect = [
            create_api_error(RateLimitError),
            create_mock_response()
        ]

        messages = [{"role": "user", "content": "Test"}]
        result = call_llm_with_retry(messages, max_retries=2)

        assert result is not None
        assert mock_client.chat.completions.create.call_count == 2
        mock_sleep.assert_called_once_with(2)  # First retry delay

    @patch('analyzer.time.sleep')
    @patch('analyzer.client')
    def test_api_connection_error_triggers_retry(self, mock_client, mock_sleep):
        """APIConnectionError should trigger retry."""
        mock_client.chat.completions.create.side_effect = [
            create_api_error(APIConnectionError),
            create_mock_response()
        ]

        messages = [{"role": "user", "content": "Test"}]
        result = call_llm_with_retry(messages, max_retries=2)

        assert result is not None
        assert mock_client.chat.completions.create.call_count == 2

    @patch('analyzer.time.sleep')
    @patch('analyzer.client')
    def test_internal_server_error_triggers_retry(self, mock_client, mock_sleep):
        """InternalServerError should trigger retry."""
        mock_client.chat.completions.create.side_effect = [
            create_api_error(InternalServerError),
            create_mock_response()
        ]

        messages = [{"role": "user", "content": "Test"}]
        result = call_llm_with_retry(messages, max_retries=2)

        assert result is not None
        assert mock_client.chat.completions.create.call_count == 2

    @patch('analyzer.time.sleep')
    @patch('analyzer.client')
    def test_internal_server_error_500_triggers_retry(self, mock_client, mock_sleep):
        """InternalServerError (500) should trigger retry."""
        mock_client.chat.completions.create.side_effect = [
            create_api_error(InternalServerError),
            create_mock_response()
        ]

        messages = [{"role": "user", "content": "Test"}]
        result = call_llm_with_retry(messages, max_retries=2)

        assert result is not None
        assert mock_client.chat.completions.create.call_count == 2


class TestCallLLMNonRetryableErrors:
    """Tests for non-retryable error handling."""

    @patch('analyzer.client')
    def test_bad_request_error_no_retry(self, mock_client):
        """BadRequestError should not trigger retry."""
        mock_client.chat.completions.create.side_effect = create_api_error(BadRequestError)

        messages = [{"role": "user", "content": "Test"}]
        with pytest.raises(Exception, match="Invalid request \\(no retry\\)"):
            call_llm_with_retry(messages, max_retries=2)

        # Should only be called once (no retries)
        assert mock_client.chat.completions.create.call_count == 1

    @patch('analyzer.client')
    def test_authentication_error_no_retry(self, mock_client):
        """AuthenticationError should not trigger retry."""
        mock_client.chat.completions.create.side_effect = create_api_error(AuthenticationError)

        messages = [{"role": "user", "content": "Test"}]
        with pytest.raises(Exception, match="Invalid request \\(no retry\\)"):
            call_llm_with_retry(messages, max_retries=2)

        assert mock_client.chat.completions.create.call_count == 1

    @patch('analyzer.client')
    def test_permission_denied_error_no_retry(self, mock_client):
        """PermissionDeniedError should not trigger retry."""
        mock_client.chat.completions.create.side_effect = create_api_error(PermissionDeniedError)

        messages = [{"role": "user", "content": "Test"}]
        with pytest.raises(Exception, match="Invalid request \\(no retry\\)"):
            call_llm_with_retry(messages, max_retries=2)

        assert mock_client.chat.completions.create.call_count == 1

    @patch('analyzer.client')
    def test_not_found_error_no_retry(self, mock_client):
        """NotFoundError should not trigger retry."""
        mock_client.chat.completions.create.side_effect = create_api_error(NotFoundError)

        messages = [{"role": "user", "content": "Test"}]
        with pytest.raises(Exception, match="Invalid request \\(no retry\\)"):
            call_llm_with_retry(messages, max_retries=2)

        assert mock_client.chat.completions.create.call_count == 1


class TestCallLLMExponentialBackoff:
    """Tests for exponential backoff timing."""

    @patch('analyzer.time.sleep')
    @patch('analyzer.client')
    def test_first_retry_delay_is_2_seconds(self, mock_client, mock_sleep):
        """First retry should wait 2 seconds."""
        mock_client.chat.completions.create.side_effect = [
            create_api_error(RateLimitError),
            create_mock_response()
        ]

        messages = [{"role": "user", "content": "Test"}]
        call_llm_with_retry(messages, max_retries=2)

        mock_sleep.assert_called_with(2)

    @patch('analyzer.time.sleep')
    @patch('analyzer.client')
    def test_second_retry_delay_is_5_seconds(self, mock_client, mock_sleep):
        """Second retry should wait 5 seconds."""
        mock_client.chat.completions.create.side_effect = [
            create_api_error(RateLimitError),
            create_api_error(RateLimitError),
            create_mock_response()
        ]

        messages = [{"role": "user", "content": "Test"}]
        call_llm_with_retry(messages, max_retries=2)

        # Check both delays were used
        calls = mock_sleep.call_args_list
        assert calls[0][0][0] == 2  # First delay
        assert calls[1][0][0] == 5  # Second delay


class TestCallLLMMaxRetriesExceeded:
    """Tests for max retries exceeded behavior."""

    @patch('analyzer.time.sleep')
    @patch('analyzer.client')
    def test_max_retries_exceeded_raises_exception(self, mock_client, mock_sleep):
        """Should raise exception after max retries exceeded."""
        mock_client.chat.completions.create.side_effect = create_api_error(RateLimitError)

        messages = [{"role": "user", "content": "Test"}]
        with pytest.raises(Exception, match="Rate limit exceeded after 2 retries"):
            call_llm_with_retry(messages, max_retries=2)

        # Initial call + 2 retries = 3 total calls
        assert mock_client.chat.completions.create.call_count == 3

    @patch('analyzer.time.sleep')
    @patch('analyzer.client')
    def test_server_error_max_retries_message(self, mock_client, mock_sleep):
        """Server error message should be returned after max retries."""
        mock_client.chat.completions.create.side_effect = create_api_error(InternalServerError)

        messages = [{"role": "user", "content": "Test"}]
        with pytest.raises(Exception, match="Server error after 2 retries"):
            call_llm_with_retry(messages, max_retries=2)


class TestCallLLMRecoveryAfterRetry:
    """Tests for successful recovery after retries."""

    @patch('analyzer.time.sleep')
    @patch('analyzer.client')
    def test_success_after_one_retry(self, mock_client, mock_sleep):
        """Should succeed after one retry."""
        mock_response = create_mock_response(content='{"claims": [{"claim": "Test"}]}')
        mock_client.chat.completions.create.side_effect = [
            create_api_error(RateLimitError),
            mock_response
        ]

        messages = [{"role": "user", "content": "Test"}]
        result = call_llm_with_retry(messages, max_retries=2)

        assert result == mock_response
        assert mock_client.chat.completions.create.call_count == 2

    @patch('analyzer.time.sleep')
    @patch('analyzer.client')
    def test_success_after_two_retries(self, mock_client, mock_sleep):
        """Should succeed after two retries."""
        mock_response = create_mock_response()
        mock_client.chat.completions.create.side_effect = [
            create_api_error(RateLimitError),
            create_api_error(APIConnectionError),
            mock_response
        ]

        messages = [{"role": "user", "content": "Test"}]
        result = call_llm_with_retry(messages, max_retries=2)

        assert result == mock_response
        assert mock_client.chat.completions.create.call_count == 3
