"""
State definitions for the Trecl graph.
This module defines the central 'memory' of the multi-agent system.
"""

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class GithubIssue(TypedDict):
    """Schema for a single scraped GitHub issue."""
    title: str
    url: str
    repo_name: str

class GithubPR(TypedDict):
    """Schema for a single scraped GitHub Pull Request."""
    title: str
    url: str
    repo_name: str
    
class OpportunityItem(TypedDict):
    """A single actionable opportunity surfaced by any research agent."""
    type: str            # "job_posting" | "github_issue" | "github_pr" | "hackathon"
    title: str           # Human-readable label for CLI display
    description: str     # Brief summary of the role/issue (JD summary, issue body, etc.)
    url: str             # Direct link to the opportunity
    source: str          # Which agent found it: "job_decoder" | "github_analyst"
    relevance: str       # One-line explanation of why this matters for the user
    action_tier: str     # "Tier 1: Active Listing" | "Tier 2: OSS Pitch" | "Tier 3: Cold Outreach"
    suggested_action: str  # Concrete next step, e.g. "Submit PR fixing issue #142"

class GithubAnalystState(TypedDict):
    """
    Isolated state for the GitHub Analyst ReAct sub-graph.
    This prevents intermediate tool calls from bloating the global TreclState.
    """
    # INPUT from main graph
    company_name: str
    
    # LOCAL ReAct Scratchpad (Isolates tool calls)
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # OUTPUT to main graph
    github_issues: list[GithubIssue]
    github_prs: list[GithubPR]

class TreclState(TypedDict):
    """
    The shared state dictionary updated by agents throughout the graph execution.
    
    Attributes:
        company_name (str): The name of the company being researched (Initial Input).
        user_domain (str): The user's technical field (e.g. "backend", "frontend").
        user_stack (list[str]): The user's specific technologies (e.g. ["Python", "Go"]).
        user_anti_persona (str): Roles or tasks to explicitly exclude (e.g., "No ML research").
        company_summary (str): The synthesized research containing 5 core facts.
            Populated by the `company_researcher_node`.
        company_jobs (str): The synthesized open roles and hiring needs.
            Populated by the `job_decoder_node`.
        github_issues (list[GithubIssue]): Valid open-source issues the candidate can solve.
            Populated by the `github_analyst_node`.
        github_prs (list[GithubPR]): Stale or open PRs the candidate could review or fix.
            Populated by the `github_analyst_node`.
        curated_opportunities (list[OpportunityItem]): The ranked list of opportunities to present.
            Populated by the `opportunity_curator_node`.
        selected_targets (list[OpportunityItem]): The specific opportunities chosen by the user.
            Populated during the HITL pause.
        pain_points_ranked (str): Detailed ranking of company's technical pain points.
            Populated by the `pain_synthesizer_node`. (Will now only use selected_targets)
        project_ideas (str): Specific custom project ideas based on the user's stack.
            Populated by the `pain_synthesizer_node`. (Will now only use selected_targets)
        cold_email (str): The generated targeted cold outreach email.
            Populated by the `cold_email_writer_node`.
    """
    company_name: str
    user_domain: str
    user_stack: list[str]
    user_anti_persona: str
    company_summary: str
    company_jobs: str
    github_issues: list[GithubIssue]
    github_prs: list[GithubPR]
    curated_opportunities: list[OpportunityItem]
    selected_targets: list[OpportunityItem]
    pain_points_ranked: str
    project_ideas: str
    cold_email: str
