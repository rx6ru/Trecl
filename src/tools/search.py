"""
Search tool wrappers utilizing the Tavily API.
Provides intelligent web-scraping and search aggregation for the researcher agents.
"""

from tavily import TavilyClient
from core.config import TAVILY_API_KEYS, USE_MOCK_SEARCH

def perform_deep_company_research(company_name: str) -> str:
    """
    Executes multiple targeted web searches to build a comprehensive profile
    of the given company, aggregating the results into a single context string.
    
    This utilizes round-robin API keys per query execution to avoid rate limits.
    If USE_MOCK_SEARCH is True, returns a structured dummy string to save API credits.
    
    Args:
        company_name (str): The name of the company to research.
        
    Returns:
        str: A concatenated block of research snippets and links.
    """
    
    if USE_MOCK_SEARCH:
        print("\n[MOCK MODE] Skipping Tavily API calls to save credits.")
        return f"""
        - What they do: {company_name} is an ultrafast grocery delivery service operating entirely through dark stores to achieve 10-minute deliveries.
        - Tech Stack: {company_name} heavily relies on PostgreSQL, Kubernetes, Backstage for developer portals, and Argo for continuous delivery.
        - Funding: {company_name} recently closed a $450m Series J funding round at a $7 Billion valuation.
        - Scale: Operating over 1,000 dark stores across major metros.
        - Engineering Challenges: Scaling data workloads to provide real-time inventory latency and managing rapid infrastructure tech debt.
        """
        
    results = []
    
    # 4 Targeted queries as defined in v0.2 plan
    queries = [
        f"{company_name} startup what they do product",
        f"{company_name} tech stack engineering blog",
        f"{company_name} funding raised 2024 2025",
        f"{company_name} engineering challenges problems"
    ]
    
    for query in queries:
        try:
            # We initialize the client per query to ensure round-robin distribution
            tavily_client = TavilyClient(api_key=TAVILY_API_KEYS.get_next_key())
            search_result = tavily_client.search(query=query, max_results=3)
            # Iterate through the results and formatted string appending
            for r in search_result.get('results', []):
                # Clean and isolate useful content snippet
                clean_content = r['content'][:300].replace('\n', ' ')
                results.append(f"- {r['title']}: {clean_content}")
        except Exception as e:
            # Degrade gracefully if a single query fails
            results.append(f"Failed to execute query '{query}': {str(e)}")
    # Combine all results into a single context block
    return "\n".join(results)

def perform_job_research(company_name: str) -> str:
    """
    Executes targeted web searches specifically to find open roles,
    hiring trends, and engineering team structure for the given company.
    
    If USE_MOCK_SEARCH is True, returns a structured dummy string.
    
    Args:
        company_name (str): The name of the company.
        
    Returns:
        str: A concatenated block of job research snippets.
    """
    if USE_MOCK_SEARCH:
        print("\n[MOCK MODE] Skipping Tavily API calls to save credits.")
        return f"""
        - Open Roles: Zepto is actively hiring Senior Backend Engineers (Python, Go) and Data Engineers.
        - Core Requirements: Needs heavy experience with scaling Kubernetes, optimizing PostgreSQL indexing, and Kafka event streaming.
        - Team Expansion: They are aggressively expanding their platform team to handle the 10x traffic spike expected in Q4.
        """
        
    results = []
    
    queries = [
        f"{company_name} careers open roles engineering",
        f"{company_name} hiring backend frontend machine learning",
        f"{company_name} engineering team structure"
    ]
    
    for query in queries:
        try:
            tavily_client = TavilyClient(api_key=TAVILY_API_KEYS.get_next_key())
            search_result = tavily_client.search(query=query, max_results=3)
            for r in search_result.get('results', []):
                clean_content = r['content'][:300].replace('\n', ' ')
                results.append(f"- {r['title']} ({r.get('url', 'N/A')}): {clean_content}")
        except Exception as e:
            results.append(f"Failed to execute query '{query}': {str(e)}")
            
    return "\n".join(results)

