"""
Health checks: Verify live API connectivity for all external services.
Requires .env to be populated with real credentials.
"""

import pytest

pytestmark = pytest.mark.health


class TestQdrantHealth:
    """Verify Qdrant Cloud connectivity."""

    def test_qdrant_connection(self):
        """QdrantClient connects and returns collections list."""
        from qdrant_client import QdrantClient
        from core.config import QDRANT_URL, QDRANT_API_KEY

        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        response = client.get_collections()
        assert response is not None
        assert hasattr(response, "collections")

    def test_qdrant_collection_exists(self):
        """The trecl_knowledge collection is present."""
        from qdrant_client import QdrantClient
        from core.config import QDRANT_URL, QDRANT_API_KEY

        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        collections = client.get_collections()
        names = [c.name for c in collections.collections]
        assert "trecl_knowledge" in names, f"Collection not found. Available: {names}"
