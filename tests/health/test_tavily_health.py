"""
Health check: Verify Tavily search API returns results.
"""

import pytest

pytestmark = pytest.mark.health


class TestTavilyHealth:
    """Verify Tavily search API connectivity."""

    def test_tavily_search_returns_results(self):
        """TavilyClient.search('test') returns at least 1 result."""
        from tavily import TavilyClient
        from core.config import TAVILY_API_KEYS

        client = TavilyClient(api_key=TAVILY_API_KEYS.get_next_key())
        response = client.search(query="Python programming language", max_results=1)

        assert "results" in response
        assert len(response["results"]) >= 1
