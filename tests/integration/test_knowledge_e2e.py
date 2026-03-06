"""
Integration test: Knowledge Store E2E against live Qdrant + Gemini.
Tests the full ingest → search → clear lifecycle.
"""

import pytest
import time

pytestmark = pytest.mark.health


class TestKnowledgeStoreE2E:
    """End-to-end tests for the knowledge store with live services."""

    # Use a unique company name to avoid polluting real data
    TEST_COMPANY = f"_test_company_{int(time.time())}"

    def test_ingest_search_clear_cycle(self):
        """Full lifecycle: ingest → search → verify → clear → verify empty."""
        from core.knowledge_store import TreclKnowledgeStore

        store = TreclKnowledgeStore()

        # 1. Ingest
        store.ingest(
            texts=[
                "TestCompany builds real-time analytics for e-commerce platforms using Kafka and ClickHouse.",
                "They are hiring Go engineers to scale their data pipeline and reduce query latency."
            ],
            metadatas=[
                {"company_name": self.TEST_COMPANY, "source_type": "website", "url": "https://test.com/1", "timestamp_epoch": int(time.time())},
                {"company_name": self.TEST_COMPANY, "source_type": "jobs", "url": "https://test.com/2", "timestamp_epoch": int(time.time())}
            ]
        )

        # 2. Search — should find results
        results = store.search(
            query="data pipeline and Kafka",
            company_name=self.TEST_COMPANY,
            top_k=5
        )
        assert len(results) > 0, "Search returned no results after ingest"

        # 3. Clear
        store.clear(self.TEST_COMPANY)

        # 4. Search again — should be empty
        # Small delay to allow Qdrant to process the delete
        time.sleep(1)
        results_after = store.search(
            query="data pipeline",
            company_name=self.TEST_COMPANY,
            top_k=5
        )
        assert len(results_after) == 0, f"Expected 0 results after clear, got {len(results_after)}"

    def test_search_company_isolation(self):
        """Company A's data is NOT returned when searching for Company B."""
        from core.knowledge_store import TreclKnowledgeStore

        store = TreclKnowledgeStore()
        company_a = f"_isolation_a_{int(time.time())}"
        company_b = f"_isolation_b_{int(time.time())}"

        try:
            store.ingest(
                texts=["Company A specializes in Kubernetes orchestration."],
                metadatas=[{"company_name": company_a, "source_type": "web", "url": "https://a.com", "timestamp_epoch": int(time.time())}]
            )

            results = store.search(
                query="Kubernetes",
                company_name=company_b,
                top_k=5
            )
            assert len(results) == 0, "Company isolation broken: B's search returned A's data"
        finally:
            store.clear(company_a)
            store.clear(company_b)

    def test_search_source_filter(self):
        """source_filter='jobs' only returns job-type chunks."""
        from core.knowledge_store import TreclKnowledgeStore

        store = TreclKnowledgeStore()
        company = f"_source_filter_{int(time.time())}"

        try:
            store.ingest(
                texts=[
                    "They are hiring Python developers for backend services.",
                    "Their blog discusses their Python microservices architecture."
                ],
                metadatas=[
                    {"company_name": company, "source_type": "jobs", "url": "https://test.com/jobs", "timestamp_epoch": int(time.time())},
                    {"company_name": company, "source_type": "blog", "url": "https://test.com/blog", "timestamp_epoch": int(time.time())}
                ]
            )

            results = store.search(
                query="Python",
                company_name=company,
                source_filter="jobs",
                top_k=5
            )
            for r in results:
                assert r.get("source_type") == "jobs", f"Unexpected source_type: {r.get('source_type')}"
        finally:
            store.clear(company)
