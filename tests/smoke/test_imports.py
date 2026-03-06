"""
Smoke tests: Verify all modules import without error.
These tests require no API keys or network access.
"""


def test_import_core_config():
    from core.config import RoundRobinKeyManager, get_required_env_list
    assert RoundRobinKeyManager is not None
    assert callable(get_required_env_list)


def test_import_state():
    from core.state import TreclState, GithubAnalystState, GithubIssue, GithubPR, OpportunityItem
    assert TreclState is not None
    assert GithubAnalystState is not None
    assert GithubIssue is not None
    assert GithubPR is not None
    assert OpportunityItem is not None


def test_import_knowledge_store_class():
    from core.knowledge_store import TreclKnowledgeStore
    assert TreclKnowledgeStore is not None


def test_import_llm():
    from llm.model import llm
    assert llm is not None


def test_import_tools_search():
    from tools.search import perform_job_research
    assert callable(perform_job_research)


def test_import_tools_github():
    from tools.github import (
        resolve_github_handle,
        reset_guardrails,
        list_org_repos,
        get_repo_stats,
        get_repo_labels,
        search_issues,
        search_prs,
        read_issue_thread
    )
    assert callable(resolve_github_handle)
    assert callable(reset_guardrails)


def test_import_agents():
    from agents import (
        data_ingester_node,
        cold_email_writer_node,
        job_decoder_node,
        pain_synthesizer_node,
        github_analyst_node,
        opportunity_curator_node
    )
    assert callable(data_ingester_node)
    assert callable(cold_email_writer_node)
    assert callable(job_decoder_node)
    assert callable(pain_synthesizer_node)
    assert callable(github_analyst_node)
    assert callable(opportunity_curator_node)
