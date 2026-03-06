"""
Unit tests: LLM retry logic.
Verifies that transient errors (429/5xx) trigger retries while permanent
errors (400, ValidationError) do not.
"""

import pytest
from llm.model import is_transient_llm_error, ChatCerebrasWithRetry


class TestIsTransientError:
    """Tests for the is_transient_llm_error helper."""

    def test_429_is_transient(self):
        """429 Too Many Requests is retryable."""
        exc = Exception("Error 429: too_many_requests_error - queue_exceeded")
        assert is_transient_llm_error(exc) is True

    def test_503_is_transient(self):
        """503 Service Unavailable is retryable."""
        exc = Exception("HTTP 503 Service Unavailable")
        assert is_transient_llm_error(exc) is True

    def test_500_is_transient(self):
        """500 Internal Server Error is retryable."""
        exc = Exception("HTTP 500 Internal Server Error")
        assert is_transient_llm_error(exc) is True

    def test_502_is_transient(self):
        """502 Bad Gateway is retryable."""
        exc = Exception("HTTP 502 Bad Gateway")
        assert is_transient_llm_error(exc) is True

    def test_504_is_transient(self):
        """504 Gateway Timeout is retryable."""
        exc = Exception("HTTP 504 Gateway Timeout")
        assert is_transient_llm_error(exc) is True

    def test_queue_exceeded_is_transient(self):
        """Cerebras queue_exceeded flag is retryable."""
        exc = Exception("queue_exceeded: High traffic, please retry later")
        assert is_transient_llm_error(exc) is True

    def test_400_is_not_transient(self):
        """400 Bad Request should NOT be retried."""
        exc = Exception("HTTP 400 Bad Request: invalid prompt format")
        assert is_transient_llm_error(exc) is False

    def test_validation_error_is_not_transient(self):
        """Pydantic ValidationError should NOT be retried."""
        exc = Exception("ValidationError: field 'name' is required")
        assert is_transient_llm_error(exc) is False

    def test_generic_exception_is_not_transient(self):
        """Random exceptions should NOT be retried."""
        exc = Exception("Something completely unexpected happened")
        assert is_transient_llm_error(exc) is False

    def test_auth_error_is_not_transient(self):
        """401 Unauthorized should NOT be retried (bad API key)."""
        exc = Exception("HTTP 401 Unauthorized: Invalid API key")
        assert is_transient_llm_error(exc) is False


class TestChatCerebrasWithRetry:
    """Tests for the retry wrapper proxy."""

    def test_invoke_delegates_to_underlying_llm(self):
        """invoke() calls the underlying LLM's invoke method."""
        from unittest.mock import MagicMock
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Hello!")
        
        wrapper = ChatCerebrasWithRetry(mock_llm)
        result = wrapper.invoke("test prompt")
        
        mock_llm.invoke.assert_called_once_with("test prompt")
        assert result.content == "Hello!"

    def test_bind_tools_returns_wrapped_instance(self):
        """bind_tools() returns a new ChatCerebrasWithRetry."""
        from unittest.mock import MagicMock
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = MagicMock()
        
        wrapper = ChatCerebrasWithRetry(mock_llm)
        bound = wrapper.bind_tools(["tool1"])
        
        assert isinstance(bound, ChatCerebrasWithRetry)
        mock_llm.bind_tools.assert_called_once_with(["tool1"])

    def test_with_structured_output_returns_invokable(self):
        """with_structured_output() returns an object with an invoke() method."""
        from unittest.mock import MagicMock
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            name: str

        mock_llm = MagicMock()
        mock_runner = MagicMock()
        mock_runner.invoke.return_value = TestSchema(name="test")
        mock_llm.with_structured_output.return_value = mock_runner

        wrapper = ChatCerebrasWithRetry(mock_llm)
        structured = wrapper.with_structured_output(TestSchema)

        assert hasattr(structured, "invoke")
        result = structured.invoke("test prompt")
        assert result.name == "test"
