"""
GitHub API Interaction Tool.
Uses PyGithub to fetch repositories, issues, and PRs. Includes a specialized
LLM-powered tool to resolve a plain company name to its exact GitHub handle.
"""

from github import Github
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from core.config import GITHUB_ACCESS_TOKEN, USE_MOCK_GITHUB
from tools.search import TavilyClient, TAVILY_API_KEYS
from llm.model import llm

class GithubHandleOutput(BaseModel):
    handle: str = Field(description="The exact GitHub organization handle, e.g. 'zeptonow'")

def resolve_github_handle(company_name: str) -> str:
    """
    Uses Tavily and the LLM to find the official GitHub organization handle
    for a given company name.
    
    Args:
        company_name (str): The plain English name of the company.
        
    Returns:
        str: The exact GitHub organization handle.
    """
    if USE_MOCK_GITHUB:
        print("\n[MOCK MODE] Skipping GitHub handle resolution.")
        return "zeptonow"
        
    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEYS.get_next_key())
        search_result = tavily_client.search(query=f"{company_name} official github url", max_results=3)
        context = str(search_result.get('results', []))
        
        prompt = f"Extract the exact GitHub organization handle for {company_name} from this search context:\n{context}"
        structured_llm = llm.with_structured_output(GithubHandleOutput)
        
        response = structured_llm.invoke([
            SystemMessage(content="You extract GitHub handles flawlessly."),
            HumanMessage(content=prompt)
        ])
        return response.handle
    except Exception as e:
        print(f"[!] Failed to resolve GitHub handle: {e}")
        return ""

def fetch_github_issues(org_handle: str) -> list[dict]:
    """
    Fetches the top 10 most-commented open issues across the org's repos.
    These reveal real architectural pain points, not just beginner-friendly tasks.
    Also searches for issues labeled 'bug' or 'enhancement' as supplementary signals.
    """
    if USE_MOCK_GITHUB:
        return [
            {"title": "Fix slow PostgreSQL indexing on delivery route query", "url": "https://github.com/zeptonow/routing-engine/issues/142", "repo_name": "routing-engine"}
        ]
        
    if not org_handle or not GITHUB_ACCESS_TOKEN:
        return []
        
    g = Github(GITHUB_ACCESS_TOKEN)
    results = []
    seen_urls = set()
    
    # Primary query: top open issues sorted by most comments (highest engagement = real pain)
    queries = [
        f"org:{org_handle} is:open is:issue sort:comments-desc",
        f"org:{org_handle} is:open is:issue label:bug",
        f"org:{org_handle} is:open is:issue label:enhancement",
    ]
    
    for query in queries:
        try:
            issues = g.search_issues(query, sort="comments", order="desc")
            # Safely iterate with bounds checking
            count = 0
            for issue in issues:
                if count >= 5:
                    break
                if issue.html_url in seen_urls:
                    continue
                seen_urls.add(issue.html_url)
                results.append({
                    "title": issue.title,
                    "url": issue.html_url,
                    "repo_name": issue.repository.name
                })
                count += 1
        except Exception as e:
            print(f"[!] GitHub issue query failed ('{query[:40]}...'): {e}")
    
    # Cap at 10 total to keep context manageable
    return results[:10]

def fetch_github_prs(org_handle: str) -> list[dict]:
    """
    Fetches up to 5 stale or unreviewed Pull Requests from the organization.
    """
    if USE_MOCK_GITHUB:
        return [
            {"title": "feat: Add distributed Redis locking for inventory sync", "url": "https://github.com/zeptonow/inventory-service/pull/89", "repo_name": "inventory-service"},
            {"title": "fix: Resolve K8s memory leak in Go worker pool", "url": "https://github.com/zeptonow/worker-pool/pull/42", "repo_name": "worker-pool"}
        ]
        
    if not org_handle or not GITHUB_ACCESS_TOKEN:
        return []
        
    g = Github(GITHUB_ACCESS_TOKEN)
    results = []
    
    # Search for open PRs that have no reviews and are not drafts
    query = f"org:{org_handle} is:open is:pr review:none draft:false"
    
    try:
        prs = g.search_issues(query, sort="updated", order="desc")
        count = 0
        for pr in prs:
            if count >= 5:
                break
            results.append({
                "title": pr.title,
                "url": pr.html_url,
                "repo_name": pr.repository.name
            })
            count += 1
    except Exception as e:
        print(f"[!] Failed to fetch GitHub PRs: {e}")
        
    return results
