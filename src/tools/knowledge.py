"""
Knowledge retrieval tools for Trecl agents.
Connects the LLM ReAct loops to the Qdrant VectorDB.
"""

from typing import Optional
from langchain_core.tools import tool
from core.knowledge_store import TreclKnowledgeStore

@tool
def search_company_knowledge(
    query: str,
    company_name: str,
    source_filter: Optional[str] = None,
    max_age_days: int = 540,
    top_k: int = 5
) -> str:
    """
    Search the company VectorDB for specific engineering context, documentation, or discussions.
    
    Args:
        query: The specific question or topic to search for (e.g., "github repositories", "core architecture", "tech stack").
        company_name: The EXACT name of the company being researched.
        source_filter: Optional strictly typed filter to restrict results. 
                       Allowed values: "company_website", "github_readme", "github_issue", 
                                       "jobs", "news", "community", "social".
        max_age_days: Maximum age of the information in days. Default is 540 (1.5 years).
        top_k: Number of chunks to retrieve. Default is 5.
        
    Returns:
        A formatted string of context chunks matching the query.
    """
    store = TreclKnowledgeStore()
    
    try:
        results = store.search(
            query=query,
            company_name=company_name,
            source_filter=source_filter,
            max_age_days=max_age_days,
            top_k=top_k
        )
        
        if not results:
            return f"No results found in knowledge base for query: '{query}'"
            
        context_blocks = []
        for r in results:
            source = r.get('source_type', 'unknown')
            url = r.get('url', 'N/A')
            content = r.get('content', '')
            context_blocks.append(f"[Source: {source} | URL: {url}]\n{content}\n")
            
        return "\n".join(context_blocks)
        
    except Exception as e:
        return f"Error querying knowledge store: {str(e)}"
