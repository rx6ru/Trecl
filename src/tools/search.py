"""
Search tool wrappers utilizing the Tavily API.
Provides intelligent web-scraping and search aggregation for the researcher agents.
"""

from tavily import TavilyClient
from core.config import TAVILY_API_KEYS, USE_MOCK_SEARCH

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
                clean_content = r['content'].replace('\n', ' ')
                results.append(f"- {r['title']} ({r.get('url', 'N/A')}): {clean_content}")
        except Exception as e:
            results.append(f"Failed to execute query '{query}': {str(e)}")
            
    return "\n".join(results)

