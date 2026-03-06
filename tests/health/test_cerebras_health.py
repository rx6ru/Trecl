"""
Health check: Verify Cerebras LLM responds to a simple prompt.
"""

import pytest

pytestmark = pytest.mark.health


class TestCerebrasHealth:
    """Verify Cerebras LLM connectivity."""

    def test_cerebras_invoke(self):
        """LLM responds to a simple prompt with non-empty content."""
        from langchain_cerebras import ChatCerebras
        from pydantic import SecretStr
        from core.config import CEREBRAS_API_KEYS

        llm = ChatCerebras(
            model="gpt-oss-120b",
            api_key=SecretStr(CEREBRAS_API_KEYS.get_next_key()),
            max_tokens=50,
        )

        response = llm.invoke("Say the word 'pong'. Nothing else.")
        assert response is not None
        assert hasattr(response, "content")
        assert len(response.content) > 0
