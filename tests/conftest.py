"""
Shared fixtures for the Trecl test suite.
Provides mock state, mock environment, and mock external services.
"""

import os
import pytest
from unittest.mock import MagicMock, patch


# ─── Mock Environment ─────────────────────────────────────────────────
# These fixtures prevent real API keys from being required during unit tests.

@pytest.fixture
def mock_env(monkeypatch):
    """Patches environment variables with dummy API keys for testing."""
    monkeypatch.setenv("CEREBRAS_API_KEY", "test-cerebras-key-1,test-cerebras-key-2")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    monkeypatch.setenv("GITHUB_ACCESS_TOKEN", "test-github-pat")
    monkeypatch.setenv("GEMINI_API_KEYS", "test-gemini-key-1,test-gemini-key-2")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "test-qdrant-key")
    monkeypatch.setenv("USE_MOCK_SEARCH", "true")
    monkeypatch.setenv("USE_MOCK_LLM", "true")
    monkeypatch.setenv("USE_MOCK_GITHUB", "true")


# ─── Mock State ───────────────────────────────────────────────────────

@pytest.fixture
def mock_trecl_state():
    """Returns a fully populated TreclState dictionary for testing."""
    return {
        "company_name": "TestCorp",
        "user_domain": "backend",
        "user_stack": ["Python", "Go", "PostgreSQL"],
        "user_anti_persona": "No ML research",
        "company_summary": "TestCorp is a fintech startup building real-time payment infrastructure.",
        "company_jobs": "Hiring Senior Backend Engineers (Go, Python).",
        "github_issues": [
            {"title": "Fix slow query on payments table", "url": "https://github.com/testcorp/api/issues/42", "repo_name": "testcorp/api"}
        ],
        "github_prs": [
            {"title": "feat: Add Redis caching layer", "url": "https://github.com/testcorp/api/pull/55", "repo_name": "testcorp/api"}
        ],
        "curated_opportunities": [],
        "selected_targets": [
            {
                "type": "github_issue",
                "title": "Fix slow query on payments table",
                "description": "PostgreSQL indexing issue causing 2s latency",
                "url": "https://github.com/testcorp/api/issues/42",
                "source": "github_analyst",
                "relevance": "Directly matches candidate's PostgreSQL expertise",
                "action_tier": "Tier 2: OSS Pitch",
                "suggested_action": "Submit PR fixing issue #42"
            }
        ],
        "pain_points_ranked": "1. Database query latency\n2. Payment processing bottleneck",
        "project_ideas": "Build a PostgreSQL query optimizer middleware",
        "cold_email": ""
    }


# ─── Mock Qdrant Client ──────────────────────────────────────────────

@pytest.fixture
def mock_qdrant_client():
    """Returns a mocked QdrantClient with pre-configured responses."""
    client = MagicMock()
    
    # Simulate collection already exists
    collection_mock = MagicMock()
    collection_mock.name = "trecl_knowledge"
    collections_response = MagicMock()
    collections_response.collections = [collection_mock]
    client.get_collections.return_value = collections_response
    
    # Simulate query_points returning empty by default
    query_response = MagicMock()
    query_response.points = []
    client.query_points.return_value = query_response
    
    return client


# ─── Mock Gemini Embeddings ───────────────────────────────────────────

@pytest.fixture
def mock_gemini_embedding():
    """Returns a fake 768-dim embedding response."""
    return {"embedding": [[0.1] * 768]}


def fake_embed_content(**kwargs):
    """Fake gemini_client.embed_content that returns correct-shape vectors."""
    content = kwargs.get("content", "")
    if isinstance(content, list):
        return {"embedding": [[0.1] * 768 for _ in content]}
    return {"embedding": [0.1] * 768}
