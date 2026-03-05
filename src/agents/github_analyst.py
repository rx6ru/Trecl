"""
GitHub Analyst Node.
Responsible for resolving the official GitHub handle of the target company,
and subsequently fetching high-signal open source data (issues and PRs)
using the GitHub API.
"""

from core.state import TreclState
from tools.github import resolve_github_handle, fetch_github_issues, fetch_github_prs

def github_analyst_node(state: TreclState) -> dict:
    """
    LangGraph node representing the GitHub Analyst Agent.
    
    Args:
        state (TreclState): The current graph state.
        
    Returns:
        dict: A dictionary update patch containing `github_issues` and `github_prs`.
    """
    company = state["company_name"]
    print(f"\n[~] Starting GitHub Analyst for {company}...")
    
    handle = resolve_github_handle(company)
    if not handle:
        print(f"[!] Could not resolve GitHub handle for {company}.")
        return {"github_issues": [], "github_prs": []}
        
    print(f"    [*] Resolved official handle: {handle}")
    
    issues = fetch_github_issues(handle)
    print(f"    [*] Found {len(issues)} approachable open issues.")
    
    prs = fetch_github_prs(handle)
    print(f"    [*] Found {len(prs)} unreviewed/stale PRs.")
    
    return {
        "github_issues": issues,
        "github_prs": prs
    }
