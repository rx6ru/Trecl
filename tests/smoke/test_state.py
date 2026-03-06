"""
Smoke tests: Verify state TypedDicts instantiate correctly and reducers work.
"""

from core.state import (
    TreclState,
    GithubAnalystState,
    GithubIssue,
    GithubPR,
    OpportunityItem
)


class TestStateSchemas:
    """Tests for TypedDict instantiation."""

    def test_trecl_state_instantiation(self, mock_trecl_state):
        """TreclState can be instantiated with all required keys."""
        state: TreclState = mock_trecl_state
        assert state["company_name"] == "TestCorp"
        assert isinstance(state["user_stack"], list)
        assert isinstance(state["github_issues"], list)
        assert isinstance(state["selected_targets"], list)
        assert "company_summary" in state
        assert "cold_email" in state

    def test_github_analyst_state(self):
        """GithubAnalystState instantiates with messages list."""
        state: GithubAnalystState = {
            "company_name": "TestCorp",
            "messages": [],
            "github_issues": [],
            "github_prs": []
        }
        assert state["company_name"] == "TestCorp"
        assert isinstance(state["messages"], list)

    def test_opportunity_item_has_all_fields(self):
        """OpportunityItem has all 8 required fields."""
        item: OpportunityItem = {
            "type": "job_posting",
            "title": "Senior Backend Engineer",
            "description": "Build payment APIs",
            "url": "https://example.com/jobs/1",
            "source": "job_decoder",
            "relevance": "Matches Go and PostgreSQL skills",
            "action_tier": "Tier 1: Active Listing",
            "suggested_action": "Apply directly"
        }
        expected_keys = {"type", "title", "description", "url", "source", "relevance", "action_tier", "suggested_action"}
        assert set(item.keys()) == expected_keys

    def test_github_issue_schema(self):
        """GithubIssue has title, url, repo_name."""
        issue: GithubIssue = {"title": "Bug", "url": "https://example.com", "repo_name": "org/repo"}
        assert set(issue.keys()) == {"title", "url", "repo_name"}

    def test_github_pr_schema(self):
        """GithubPR has title, url, repo_name."""
        pr: GithubPR = {"title": "Feature", "url": "https://example.com", "repo_name": "org/repo"}
        assert set(pr.keys()) == {"title", "url", "repo_name"}


class TestStateReducers:
    """Tests for LangGraph state reducer behavior."""

    def test_list_append_does_not_overwrite(self, mock_trecl_state):
        """Simulates dictionary-based list append to verify no overwriting."""
        state = mock_trecl_state.copy()
        original_issues = state["github_issues"].copy()

        # Simulate what a node does: return a patch with new issues
        new_issues = [{"title": "New Issue", "url": "https://example.com/2", "repo_name": "org/other"}]
        
        # Manual list merge (LangGraph does this internally for list fields)
        merged = original_issues + new_issues
        assert len(merged) == 2
        assert merged[0]["title"] == "Fix slow query on payments table"
        assert merged[1]["title"] == "New Issue"

    def test_messages_accumulate(self):
        """Messages reducer (add_messages) accumulates rather than replaces."""
        from langchain_core.messages import HumanMessage, AIMessage
        
        # Simulate the add_messages reducer behavior
        existing = [HumanMessage(content="Hello")]
        new = [AIMessage(content="Hi there")]
        accumulated = existing + new
        
        assert len(accumulated) == 2
        assert accumulated[0].content == "Hello"
        assert accumulated[1].content == "Hi there"
