"""
GitHub API Interaction Tool.
Uses PyGithub to fetch repositories, issues, and PRs. Includes a specialized
LLM-powered tool to resolve a plain company name to its exact GitHub handle.

Guardrails:
  - _discovered_repos: Tracks repos confirmed via list_org_repos.
  - _label_cache: Tracks repos that have had get_repo_labels called.
  - search_issues rejects label-filtered queries if labels haven't been discovered,
    and rejects repo_name if the repo hasn't been discovered via list_org_repos.
"""

from github import Github
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from typing import Optional, List

from core.config import GITHUB_ACCESS_TOKEN, USE_MOCK_GITHUB
from tools.search import TavilyClient, TAVILY_API_KEYS
from llm.model import llm

# ─── Stateful Guardrails ─────────────────────────────────────────────
# These module-level registries enforce the correct tool-calling sequence.
# The agent MUST call list_org_repos before search_issues, and
# get_repo_labels before search_issues with labels.

_org_discovered: bool = False          # Has list_org_repos been called at all?
_discovered_repos: set[str] = set()    # Repos confirmed via list_org_repos
_label_cache: dict[str, list[str]] = {}  # Repo -> list of known label names

# Operators the LLM must NOT smuggle into search_query
_BANNED_OPERATORS = ["label:", "repo:", "is:", "org:", "user:", "review:", "draft:"]

def reset_guardrails():
    """Reset guardrail state between pipeline runs."""
    global _org_discovered
    _org_discovered = False
    _discovered_repos.clear()
    _label_cache.clear()

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

@tool
def search_issues(
    repo_name: str,
    state: str = "open",
    labels: Optional[List[str]] = None,
    sort_by: str = "created",
    sort_direction: str = "desc",
    limit: int = 10,
    search_query: Optional[str] = None
) -> list[dict]:
    """
    Searches for issues in a specific GitHub repository. Use this tool to find active pain points,
    feature requests, bugs, or contribution opportunities.

    Args:
        repo_name: The full name of the repository (e.g., "hwchase17/langchain" or "vercel/next.js").
        state: The state of the issues to retrieve. Must be "open", "closed", or "all". Defaults to "open".
        labels: A list of specific label names to filter by (e.g., ["bug", "help wanted"]). Use get_repo_labels first if you do not know the exact label names.
        sort_by: The metric to sort the results by. Must be "created", "updated", or "comments". Defaults to "created". Use "comments" to find the most discussed/painful issues.
        sort_direction: The direction of the sort. Must be "asc" or "desc". Defaults to "desc".
        limit: The maximum number of issues to return. Keep this low (5-10) to avoid overloading your context window. Defaults to 10.
        search_query: Optional keyword search string to find specific topics within the repo's issues (e.g., "memory leak" or "postgres").
    """
    if not GITHUB_ACCESS_TOKEN:
        return [{"error": "GITHUB_ACCESS_TOKEN is not set."}]
    
    # GUARDRAIL 1: Reject if list_org_repos has never been called
    if not _org_discovered:
        return [{
            "error": "REJECTED: You have not called list_org_repos yet. "
                     "You MUST call list_org_repos first to discover valid repositories "
                     "before searching for issues."
        }]
    
    # GUARDRAIL 1b: Reject repo_name if it wasn't in the discovered set
    if repo_name not in _discovered_repos:
        return [{
            "error": f"REJECTED: '{repo_name}' was not found by list_org_repos. "
                     f"Do NOT hallucinate repository names. "
                     f"Known repos: {list(_discovered_repos)[:5]}"
        }]
    
    # GUARDRAIL 2: Reject label queries if get_repo_labels hasn't been called
    if labels and repo_name not in _label_cache:
        return [{
            "error": f"REJECTED: You passed labels={labels} but have not called "
                     f"get_repo_labels('{repo_name}') yet. Call it first to discover "
                     f"the exact label names this repo uses."
        }]
    
    # GUARDRAIL 3: Reject labels that don't actually exist in the repo
    if labels and repo_name in _label_cache:
        known = _label_cache[repo_name]
        invalid = [l for l in labels if l not in known]
        if invalid:
            return [{
                "error": f"REJECTED: Labels {invalid} do not exist in '{repo_name}'. "
                         f"Valid labels include: {known[:10]}. Use only exact names."
            }]
    
    # GUARDRAIL 4: Sanitize search_query — block operator injection
    if search_query:
        for op in _BANNED_OPERATORS:
            if op in search_query.lower():
                return [{
                    "error": f"REJECTED: search_query contains banned operator '{op}'. "
                             f"Use the dedicated function arguments (labels, state) instead. "
                             f"search_query is for plain keywords only (e.g., 'memory leak')."
                }]
        
    g = Github(GITHUB_ACCESS_TOKEN)
    
    # 1. Base query parts
    query_parts = [f"repo:{repo_name}", "is:issue"]
    
    # 2. Add state filter
    if state in ["open", "closed"]:
        query_parts.append(f"is:{state}")
        
    # 3. Add label filters
    if labels:
        for label in labels:
            # Wrap in quotes if the label has spaces
            safe_label = f'"{label}"' if ' ' in label else label
            query_parts.append(f"label:{safe_label}")
            
    # 4. Add keyword search
    if search_query:
        query_parts.append(search_query)
        
    # Combine into the final GitHub search string
    final_query = " ".join(query_parts)
    print(f"[*] Tool 'search_issues' querying: {final_query}")
    
    try:
        results = []
        # PyGithub uses 'order' instead of 'sort_direction'
        issues = g.search_issues(query=final_query, sort=sort_by, order=sort_direction)
        
        count = 0
        # Iterate safely
        for issue in issues:
            if count >= limit:
                break
                
            results.append({
                "number": issue.number,
                "title": issue.title,
                "state": issue.state,
                "created_at": issue.created_at.isoformat() if issue.created_at else None,
                "comments_count": issue.comments,
                "url": issue.html_url,
                "labels": [lbl.name for lbl in issue.labels],
                "body_preview": issue.body[:250] + "..." if issue.body else ""
            })
            count += 1
            
        return results
    except Exception as e:
        return [{"error": f"Failed to search GitHub issues: {str(e)}", "query_used": final_query}]


@tool
def search_prs(
    repo_name: str,
    state: str = "open",
    sort_by: str = "created",
    sort_direction: str = "desc",
    limit: int = 5,
    search_query: Optional[str] = None
) -> list[dict]:
    """
    Searches for Pull Requests in a specific GitHub repository.
    Use this tool to find unreviewed, stale, or community PRs that a candidate could
    review, improve, or build upon. This is separate from search_issues — issues and PRs
    are different GitHub objects.

    Args:
        repo_name: The full name of the repository (e.g., "traceroot-ai/traceroot").
        state: Must be "open", "closed", or "all". Defaults to "open".
        sort_by: Must be "created", "updated", or "comments". Defaults to "created". Use "updated" to find stale PRs.
        sort_direction: Must be "asc" or "desc". Defaults to "desc". Use "asc" with sort_by="updated" to find the most neglected PRs.
        limit: Maximum number of PRs to return. Keep low (3-5). Defaults to 5.
        search_query: Optional keyword search (e.g., "helm" or "docker").
    """
    if not GITHUB_ACCESS_TOKEN:
        return [{"error": "GITHUB_ACCESS_TOKEN is not set."}]

    # GUARDRAIL: Reject if list_org_repos has never been called
    if not _org_discovered:
        return [{
            "error": "REJECTED: You have not called list_org_repos yet. "
                     "You MUST call list_org_repos first to discover valid repositories."
        }]
    
    # GUARDRAIL: Reject repo_name if it wasn't in the discovered set
    if repo_name not in _discovered_repos:
        return [{
            "error": f"REJECTED: '{repo_name}' was not found by list_org_repos. "
                     f"Known repos: {list(_discovered_repos)[:5]}"
        }]
    
    # GUARDRAIL: Sanitize search_query — block operator injection
    if search_query:
        for op in _BANNED_OPERATORS:
            if op in search_query.lower():
                return [{
                    "error": f"REJECTED: search_query contains banned operator '{op}'. "
                             f"search_query is for plain keywords only."
                }]

    g = Github(GITHUB_ACCESS_TOKEN)

    # Build query — PRs use "is:pr" instead of "is:issue"
    query_parts = [f"repo:{repo_name}", "is:pr"]

    if state in ["open", "closed"]:
        query_parts.append(f"is:{state}")

    # Filter for unreviewed PRs (highest signal for contribution opportunities)
    query_parts.append("review:none")
    query_parts.append("draft:false")

    if search_query:
        query_parts.append(search_query)

    final_query = " ".join(query_parts)
    print(f"[*] Tool 'search_prs' querying: {final_query}")

    try:
        results = []
        prs = g.search_issues(query=final_query, sort=sort_by, order=sort_direction)

        count = 0
        for pr in prs:
            if count >= limit:
                break

            results.append({
                "number": pr.number,
                "title": pr.title,
                "url": pr.html_url,
                "repo_name": repo_name,
                "created_at": pr.created_at.isoformat() if pr.created_at else None,
                "comments_count": pr.comments,
                "body_preview": pr.body[:200] + "..." if pr.body and len(pr.body) > 200 else (pr.body or "")
            })
            count += 1

        return results
    except Exception as e:
        return [{"error": f"Failed to search GitHub PRs: {str(e)}", "query_used": final_query}]


@tool
def get_repo_labels(repo_name: str) -> list[dict]:
    """
    Lists all labels used in a GitHub repository, sorted by how many open issues use each label.
    Call this BEFORE search_issues to discover the exact label names a repository uses.
    Many repositories use custom labels (e.g., "area/api", "P0", "type:bug") instead of
    GitHub defaults. Using this tool prevents you from guessing labels that don't exist.

    Args:
        repo_name: The full name of the repository (e.g., "langchain-ai/langchain" or "vercel/next.js").

    Returns:
        A list of dicts with keys "name" (the exact label string) and "open_issues" (number of
        open issues using this label). Sorted by open_issues descending. Capped at 30 labels.
    """
    if not GITHUB_ACCESS_TOKEN:
        return [{"error": "GITHUB_ACCESS_TOKEN is not set."}]

    g = Github(GITHUB_ACCESS_TOKEN)

    try:
        repo = g.get_repo(repo_name)
        labels_data = []

        for label in repo.get_labels():
            # Count open issues for this label via a lightweight search
            count_query = f"repo:{repo_name} is:issue is:open label:\"{label.name}\""
            count = g.search_issues(count_query).totalCount

            labels_data.append({
                "name": label.name,
                "open_issues": count
            })

        # Sort by open issue count descending — most active labels first
        labels_data.sort(key=lambda x: x["open_issues"], reverse=True)

        # Register labels in the guardrail cache
        _label_cache[repo_name] = [l["name"] for l in labels_data]

        # Cap at 30 to protect the context window
        return labels_data[:30]

    except Exception as e:
        return [{"error": f"Failed to fetch labels for {repo_name}: {str(e)}"}]


# ─── Noise patterns to filter from issue threads ─────────────────────
# These are common low-signal comments that burn tokens without adding reasoning value.
_NOISE_PATTERNS = [
    "+1", "me too", "same here", "same issue", "any update",
    "any progress", "bump", "following", "subscribe", "is this fixed",
    "when will this be fixed", "still an issue", "still broken",
    "any eta", "please fix", "need this", "waiting for this",
]


@tool
def read_issue_thread(
    issue_url: str,
    max_comments: int = 10,
    body_char_limit: int = 500,
    comment_char_limit: int = 300
) -> dict:
    """
    Reads a GitHub issue and its most substantive comments. Use this tool to understand
    the real problem behind an issue — titles are often misleading, but the conversation
    reveals the actual technical challenge, attempted solutions, and blockers.

    This tool automatically filters out noise comments ("+1", "any updates?", short reactions)
    and prioritizes comments from repository maintainers and contributors.

    Args:
        issue_url: The full GitHub issue URL (e.g., "https://github.com/vercel/next.js/issues/12345").
        max_comments: Maximum number of filtered comments to return. Keep low (5-10) to save tokens. Defaults to 10.
        body_char_limit: Maximum characters for the issue body. Defaults to 500.
        comment_char_limit: Maximum characters per comment body. Defaults to 300.
    """
    if not GITHUB_ACCESS_TOKEN:
        return {"error": "GITHUB_ACCESS_TOKEN is not set."}

    g = Github(GITHUB_ACCESS_TOKEN)

    try:
        # Parse "owner/repo" and issue number from the URL
        # Expected format: https://github.com/{owner}/{repo}/issues/{number}
        parts = issue_url.rstrip("/").split("/")
        repo_full_name = f"{parts[-4]}/{parts[-3]}"
        issue_number = int(parts[-1])

        repo = g.get_repo(repo_full_name)
        issue = repo.get_issue(number=issue_number)

        # ─── Build the issue header ──────────────────────────────
        truncated_body = issue.body[:body_char_limit] + "..." if issue.body and len(issue.body) > body_char_limit else (issue.body or "")

        result = {
            "number": issue.number,
            "title": issue.title,
            "state": issue.state,
            "created_at": issue.created_at.isoformat() if issue.created_at else None,
            "labels": [lbl.name for lbl in issue.labels],
            "total_comments": issue.comments,
            "body": truncated_body,
            "filtered_comments": []
        }

        if issue.comments == 0:
            return result

        # ─── Fetch and filter comments ───────────────────────────
        raw_comments = list(issue.get_comments())

        scored_comments = []
        for comment in raw_comments:
            body = (comment.body or "").strip()

            # FILTER 1: Skip very short comments (reactions, "+1", etc.)
            if len(body) < 40:
                continue

            # FILTER 2: Skip comments matching known noise patterns
            body_lower = body.lower()
            if any(noise in body_lower for noise in _NOISE_PATTERNS):
                continue

            # SCORE: Prioritize by signal value
            score = 0

            # Maintainer/contributor signal — their comments carry the most weight
            author_association = getattr(comment, "author_association", "NONE")
            if author_association in ["OWNER", "MEMBER", "COLLABORATOR"]:
                score += 50
            elif author_association == "CONTRIBUTOR":
                score += 30

            # Reaction signal — community endorsed this comment
            reactions = getattr(comment, "reactions", None)
            if reactions:
                total_reactions = getattr(reactions, "total_count", 0)
                score += total_reactions * 5

            # Length signal — longer, substantive comments tend to have more technical content
            # (but diminishing returns after 500 chars)
            score += min(len(body), 500) // 50

            scored_comments.append({
                "score": score,
                "author": comment.user.login if comment.user else "unknown",
                "author_role": author_association,
                "body": body[:comment_char_limit] + "..." if len(body) > comment_char_limit else body,
                "created_at": comment.created_at.isoformat() if comment.created_at else None,
            })

        # Sort by score descending — most valuable comments first
        scored_comments.sort(key=lambda x: x["score"], reverse=True)

        # Cap at max_comments
        for comment in scored_comments[:max_comments]:
            # Drop the internal score from the output — the LLM doesn't need it
            result["filtered_comments"].append({
                "author": comment["author"],
                "author_role": comment["author_role"],
                "body": comment["body"],
                "created_at": comment["created_at"],
            })

        return result

    except Exception as e:
        return {"error": f"Failed to read issue thread: {str(e)}", "url": issue_url}


@tool
def get_repo_stats(repo_name: str) -> dict:
    """
    Fetches high-level health and technology metrics for a GitHub repository.
    Use this tool to quickly determine if a repository is active, abandoned,
    highly popular, and what programming languages it uses.
    A high open_prs_count relative to open_issues_count may indicate that
    maintainers are not reviewing community contributions — avoid pitching
    contributions to such repositories.
    
    Args:
        repo_name: The full name of the repository (e.g., "fastapi/fastapi").
        
    Returns:
        A dictionary containing the repo's description, primary languages, star count,
        fork count, open issues count, open PR count, and the timestamp of the last code push.
    """
    if not GITHUB_ACCESS_TOKEN:
        return {"error": "GITHUB_ACCESS_TOKEN is not set."}

    g = Github(GITHUB_ACCESS_TOKEN)
    
    try:
        repo = g.get_repo(repo_name)
        
        # Get language breakdown (returns dict of {Language: Bytes})
        # We only need the top 5 to save tokens
        langs = repo.get_languages()
        sorted_langs = sorted(langs.items(), key=lambda x: x[1], reverse=True)
        top_langs = [lang[0] for lang in sorted_langs[:5]]
        
        # Count open PRs — detects the "PR Black Hole" anti-pattern
        open_prs = g.search_issues(f"repo:{repo_name} is:pr is:open").totalCount
        
        return {
            "name": repo.full_name,
            "description": repo.description or "No description provided.",
            "languages": top_langs,
            "stars": repo.stargazers_count,
            "forks": repo.forks_count,
            "open_issues_count": repo.open_issues_count,
            "open_prs_count": open_prs,
            "last_pushed_at": repo.pushed_at.isoformat() if repo.pushed_at else None,
            "archived": repo.archived
        }
        
    except Exception as e:
        return {"error": f"Failed to fetch stats for {repo_name}: {str(e)}"}


@tool
def list_org_repos(org_handle: str, limit: int = 10) -> list[dict]:
    """
    Lists the top repositories in a GitHub organization, sorted by star count.
    Use this tool FIRST when you only have an organization handle (e.g., "vercel")
    and need to discover which specific repositories to investigate with
    get_repo_stats, get_repo_labels, or search_issues.

    Args:
        org_handle: The GitHub organization handle (e.g., "vercel", "traceroot-ai", "langchain-ai").
        limit: Maximum number of repositories to return. Defaults to 10. Keep low to save tokens.

    Returns:
        A list of dicts, each containing the repo's full name, star count,
        short description, and last push timestamp. Sorted by stars descending.
    """
    if not GITHUB_ACCESS_TOKEN:
        return [{"error": "GITHUB_ACCESS_TOKEN is not set."}]

    g = Github(GITHUB_ACCESS_TOKEN)

    try:
        org = g.get_organization(org_handle)
        repos = org.get_repos(type="public", sort="stars", direction="desc")

        results = []
        count = 0
        global _org_discovered
        _org_discovered = True
        for repo in repos:
            if count >= limit:
                break
            # Skip archived repos — they're read-only and not worth investigating
            if repo.archived:
                continue
            # Register in the discovery guardrail
            _discovered_repos.add(repo.full_name)
            results.append({
                "full_name": repo.full_name,
                "stars": repo.stargazers_count,
                "description": (repo.description or "")[:120],
                "last_pushed_at": repo.pushed_at.isoformat() if repo.pushed_at else None,
            })
            count += 1

        return results

    except Exception as e:
        # Fallback: maybe it's a user account, not an org
        try:
            user = g.get_user(org_handle)
            repos = user.get_repos(type="public", sort="stars", direction="desc")

            results = []
            count = 0
            _org_discovered = True  # Also set in the fallback (user account) path
            for repo in repos:
                if count >= limit:
                    break
                if repo.archived:
                    continue
                # Register in the discovery guardrail (fallback path)
                _discovered_repos.add(repo.full_name)
                results.append({
                    "full_name": repo.full_name,
                    "stars": repo.stargazers_count,
                    "description": (repo.description or "")[:120],
                    "last_pushed_at": repo.pushed_at.isoformat() if repo.pushed_at else None,
                })
                count += 1

            return results
        except Exception as e2:
            return [{"error": f"Failed to list repos for '{org_handle}': {str(e2)}"}]
