"""
Unit tests: GitHub tool guardrails.
Verifies all 4 guardrail rejection paths and the reset mechanism.
"""

import pytest
from tools.github import (
    search_issues,
    search_prs,
    reset_guardrails,
    _discovered_repos,
    _label_cache,
)
import tools.github as github_module


class TestGuardrailRejections:
    """Tests for the 4-layer guardrail defense system."""

    def setup_method(self):
        """Reset guardrails before each test."""
        reset_guardrails()

    def test_search_issues_rejects_without_org_discovery(self):
        """search_issues rejects if list_org_repos has never been called."""
        result = search_issues.invoke({"repo_name": "testcorp/api"})
        assert isinstance(result, list)
        assert "REJECTED" in result[0].get("error", "")
        assert "list_org_repos" in result[0]["error"]

    def test_search_issues_rejects_unknown_repo(self):
        """search_issues rejects repos not in _discovered_repos."""
        # Simulate org discovery has happened
        github_module._org_discovered = True
        _discovered_repos.add("testcorp/real-repo")

        result = search_issues.invoke({"repo_name": "testcorp/hallucinated-repo"})
        assert "REJECTED" in result[0].get("error", "")
        assert "hallucinated-repo" in result[0]["error"]

    def test_search_issues_rejects_unknown_labels(self):
        """search_issues rejects labels not in _label_cache."""
        github_module._org_discovered = True
        _discovered_repos.add("testcorp/api")
        # No labels cached for testcorp/api

        result = search_issues.invoke({
            "repo_name": "testcorp/api",
            "labels": ["nonexistent-label"]
        })
        assert "REJECTED" in result[0].get("error", "")
        assert "get_repo_labels" in result[0]["error"]

    def test_search_issues_rejects_invalid_labels(self):
        """search_issues rejects labels that exist in cache but weren't discovered."""
        github_module._org_discovered = True
        _discovered_repos.add("testcorp/api")
        _label_cache["testcorp/api"] = ["bug", "enhancement", "help wanted"]

        result = search_issues.invoke({
            "repo_name": "testcorp/api",
            "labels": ["P0"]  # not in the cache
        })
        assert "REJECTED" in result[0].get("error", "")
        assert "P0" in result[0]["error"]

    def test_search_issues_rejects_operator_injection(self):
        """search_issues blocks banned operators in search_query."""
        github_module._org_discovered = True
        _discovered_repos.add("testcorp/api")

        banned_queries = [
            "label:bug memory leak",
            "repo:other/repo hack",
            "is:closed old stuff",
            "org:evil injected",
        ]
        for query in banned_queries:
            result = search_issues.invoke({
                "repo_name": "testcorp/api",
                "search_query": query
            })
            assert "REJECTED" in result[0].get("error", ""), f"Failed to reject: {query}"

    def test_search_prs_rejects_without_org_discovery(self):
        """search_prs also enforces the org discovery guardrail."""
        result = search_prs.invoke({"repo_name": "testcorp/api"})
        assert isinstance(result, list)
        assert "REJECTED" in result[0].get("error", "")

    def test_search_prs_rejects_operator_injection(self):
        """search_prs blocks banned operators too."""
        github_module._org_discovered = True
        _discovered_repos.add("testcorp/api")

        result = search_prs.invoke({
            "repo_name": "testcorp/api",
            "search_query": "label:bug test"
        })
        assert "REJECTED" in result[0].get("error", "")


class TestResetGuardrails:
    """Tests for the reset_guardrails function."""

    def test_reset_clears_all_state(self):
        """reset_guardrails clears org flag, repos, and label cache."""
        github_module._org_discovered = True
        _discovered_repos.add("testcorp/api")
        _label_cache["testcorp/api"] = ["bug"]

        reset_guardrails()

        assert github_module._org_discovered is False
        assert len(_discovered_repos) == 0
        assert len(_label_cache) == 0
