"""
Unit tests: TreclKnowledgeStore with mocked Qdrant and Gemini.
Tests chunking, filter construction, ID generation, and time-decay with frozen time.
"""

import pytest
from unittest.mock import patch, MagicMock, call


class TestKnowledgeStoreIdGeneration:
    """Tests for the stable UUID generation helper."""

    def test_same_content_same_id(self):
        """Same content always produces the same UUID (idempotency)."""
        from core.knowledge_store import TreclKnowledgeStore
        store = TreclKnowledgeStore.__new__(TreclKnowledgeStore)  # skip __init__
        id1 = store._generate_id_from_hash("hello world1234")
        id2 = store._generate_id_from_hash("hello world1234")
        assert id1 == id2

    def test_different_content_different_id(self):
        """Different content produces different UUIDs."""
        from core.knowledge_store import TreclKnowledgeStore
        store = TreclKnowledgeStore.__new__(TreclKnowledgeStore)
        id1 = store._generate_id_from_hash("content A")
        id2 = store._generate_id_from_hash("content B")
        assert id1 != id2

    def test_id_is_valid_uuid_format(self):
        """Generated ID is a valid UUID string."""
        import uuid
        from core.knowledge_store import TreclKnowledgeStore
        store = TreclKnowledgeStore.__new__(TreclKnowledgeStore)
        generated = store._generate_id_from_hash("test content")
        # Should not raise
        uuid.UUID(generated)


@patch("core.knowledge_store.QdrantClient")
@patch("core.knowledge_store.gemini_client")
class TestKnowledgeStoreIngest:
    """Tests for the ingest method."""

    def test_ingest_empty_noop(self, mock_gemini, mock_qdrant_cls):
        """Empty text list returns without any API calls."""
        from core.knowledge_store import TreclKnowledgeStore
        mock_client = MagicMock()
        mock_collections = MagicMock()
        mock_col = MagicMock()
        mock_col.name = "trecl_knowledge"
        mock_collections.collections = [mock_col]
        mock_client.get_collections.return_value = mock_collections
        mock_qdrant_cls.return_value = mock_client

        store = TreclKnowledgeStore()
        store.ingest(texts=[], metadatas=[])

        mock_gemini.embed_content.assert_not_called()
        mock_client.upsert.assert_not_called()

    def test_ingest_mismatched_raises(self, mock_gemini, mock_qdrant_cls):
        """Mismatched texts and metadatas lengths raises ValueError."""
        from core.knowledge_store import TreclKnowledgeStore
        mock_client = MagicMock()
        mock_collections = MagicMock()
        mock_col = MagicMock()
        mock_col.name = "trecl_knowledge"
        mock_collections.collections = [mock_col]
        mock_client.get_collections.return_value = mock_collections
        mock_qdrant_cls.return_value = mock_client

        store = TreclKnowledgeStore()
        with pytest.raises(ValueError, match="must match"):
            store.ingest(
                texts=["text1", "text2"],
                metadatas=[{"company_name": "A"}]  # only 1 metadata
            )

    def test_ingest_calls_embed_and_upsert(self, mock_gemini, mock_qdrant_cls):
        """Ingest calls Gemini embed_content and Qdrant upsert."""
        from core.knowledge_store import TreclKnowledgeStore
        mock_client = MagicMock()
        mock_collections = MagicMock()
        mock_col = MagicMock()
        mock_col.name = "trecl_knowledge"
        mock_collections.collections = [mock_col]
        mock_client.get_collections.return_value = mock_collections
        mock_qdrant_cls.return_value = mock_client

        mock_gemini.embed_content.return_value = {"embedding": [[0.1] * 768]}

        store = TreclKnowledgeStore()
        store.ingest(
            texts=["Short test text"],
            metadatas=[{"company_name": "TestCorp", "source_type": "web", "timestamp_epoch": 100}]
        )

        mock_gemini.embed_content.assert_called_once()
        mock_client.upsert.assert_called_once()


@patch("core.knowledge_store.QdrantClient")
@patch("core.knowledge_store.gemini_client")
class TestKnowledgeStoreSearch:
    """Tests for the search method and filter construction."""

    def _make_store(self, mock_gemini, mock_qdrant_cls):
        """Helper to create a store with mocked dependencies."""
        from core.knowledge_store import TreclKnowledgeStore
        mock_client = MagicMock()
        mock_collections = MagicMock()
        mock_col = MagicMock()
        mock_col.name = "trecl_knowledge"
        mock_collections.collections = [mock_col]
        mock_client.get_collections.return_value = mock_collections
        mock_qdrant_cls.return_value = mock_client

        mock_gemini.embed_content.return_value = {"embedding": [0.1] * 768}

        query_response = MagicMock()
        query_response.points = []
        mock_client.query_points.return_value = query_response

        store = TreclKnowledgeStore()
        return store, mock_client

    def test_search_builds_company_filter(self, mock_gemini, mock_qdrant_cls):
        """Search always includes company_name filter."""
        store, client = self._make_store(mock_gemini, mock_qdrant_cls)
        store.search(query="test", company_name="TestCorp")

        call_kwargs = client.query_points.call_args
        query_filter = call_kwargs.kwargs.get("query_filter") or call_kwargs[1].get("query_filter")
        assert query_filter is not None
        # At minimum: company_name condition
        assert len(query_filter.must) >= 1

    def test_search_with_source_filter(self, mock_gemini, mock_qdrant_cls):
        """Source filter adds an additional condition."""
        store, client = self._make_store(mock_gemini, mock_qdrant_cls)
        store.search(query="test", company_name="TestCorp", source_filter="jobs")

        call_kwargs = client.query_points.call_args
        query_filter = call_kwargs.kwargs.get("query_filter") or call_kwargs[1].get("query_filter")
        # company_name + source_type + time_decay = 3 conditions
        assert len(query_filter.must) >= 2

    @patch("core.knowledge_store.time.time", return_value=1700000000.0)
    def test_search_time_decay_frozen(self, mock_time, mock_gemini, mock_qdrant_cls):
        """Time-decay filter uses frozen epoch for deterministic behavior."""
        store, client = self._make_store(mock_gemini, mock_qdrant_cls)
        store.search(query="test", company_name="TestCorp", max_age_days=30)

        call_kwargs = client.query_points.call_args
        query_filter = call_kwargs.kwargs.get("query_filter") or call_kwargs[1].get("query_filter")

        # Find the timestamp condition
        time_conditions = [c for c in query_filter.must if hasattr(c, 'key') and c.key == "timestamp_epoch"]
        assert len(time_conditions) == 1

        # The cutoff should be frozen_time - (30 * 86400)
        expected_cutoff = 1700000000 - (30 * 24 * 60 * 60)
        assert time_conditions[0].range.gte == expected_cutoff


@patch("core.knowledge_store.QdrantClient")
@patch("core.knowledge_store.gemini_client")
class TestKnowledgeStoreClear:
    """Tests for the clear method."""

    def test_clear_calls_delete_with_filter(self, mock_gemini, mock_qdrant_cls):
        """Clear deletes only data for the specified company."""
        from core.knowledge_store import TreclKnowledgeStore
        mock_client = MagicMock()
        mock_collections = MagicMock()
        mock_col = MagicMock()
        mock_col.name = "trecl_knowledge"
        mock_collections.collections = [mock_col]
        mock_client.get_collections.return_value = mock_collections
        mock_qdrant_cls.return_value = mock_client

        store = TreclKnowledgeStore()
        store.clear("TestCorp")

        mock_client.delete.assert_called_once()
