"""
Health check: Verify Gemini embedding API returns correctly shaped vectors.
"""

import pytest

pytestmark = pytest.mark.health


class TestGeminiHealth:
    """Verify Gemini embedding API connectivity."""

    def test_gemini_embed_returns_768_dim(self):
        """embed_content('hello') returns exactly a 768-dim vector."""
        import google.generativeai as gemini_client
        from core.config import GEMINI_API_KEYS

        gemini_client.configure(api_key=GEMINI_API_KEYS.get_next_key())

        response = gemini_client.embed_content(
            model="models/gemini-embedding-001",
            content="hello world",
            task_type="retrieval_query",
            output_dimensionality=768
        )

        embedding = response.get("embedding")
        assert embedding is not None, "No 'embedding' key in response"
        assert len(embedding) == 768, f"Expected 768 dims, got {len(embedding)}"
