"""
Data Ingester Node.
Phase 1 of RAG: Scrape -> Index -> Synthesize.
Blocks the fan-out until the VectorDB is populated with deep company context.
"""

import time
import sys
from typing import Dict, Any, List
from tavily import TavilyClient
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.state import TreclState
from core.config import TAVILY_API_KEYS, USE_MOCK_SEARCH
from core.knowledge_store import TreclKnowledgeStore
from llm.model import ChatCerebrasWithRetry, is_transient_llm_error
from langchain_cerebras import ChatCerebras
from pydantic import SecretStr
from core.config import CEREBRAS_API_KEYS

def _get_llm():
    """Helper to get retrying LLM."""
    base_llm = ChatCerebras(
        model="gpt-oss-120b",
        api_key=SecretStr(CEREBRAS_API_KEYS.get_next_key())
    )
    return ChatCerebrasWithRetry(base_llm)

def _classify_source(url: str, default_type: str) -> str:
    """Refine source_type based on the actual URL returned."""
    url_lower = url.lower()
    if "github.com" in url_lower:
        return "github_readme"
    if "ycombinator.com" in url_lower or "reddit.com" in url_lower or "news.ycombinator" in url_lower:
        return "community"
    if "techcrunch.com" in url_lower or "bloomberg" in url_lower:
        return "news"
    if "linkedin.com" in url_lower or "twitter.com" in url_lower or "x.com" in url_lower:
        return "social"
    if "careers" in url_lower or "jobs" in url_lower or "lever.co" in url_lower or "greenhouse.io" in url_lower:
        return "jobs"
    return default_type

def data_ingester_node(state: TreclState) -> Dict[str, Any]:
    """
    Executes the 3-phase ingestion pipeline:
    1. Scrape: Tavily deep search for diverse sources.
    2. Index: Chunk and embed into Qdrant.
    3. Synthesize: Generate company_summary from DB.
    """
    company_name = state["company_name"]
    print(f"\n[*] Starting Data Ingestion Pipeline for {company_name}...")
    
    # ----------------------------------------------------
    # Phase A: SCRAPE
    # ----------------------------------------------------
    scraped_data: List[Dict[str, Any]] = []
    
    if USE_MOCK_SEARCH:
        print("    [~] MOCK MODE: Using synthetic scraped data.")
        scraped_data = [
            {
                "content": f"{company_name} is an ultrafast grocery delivery service operating entirely through dark stores to achieve 10-minute deliveries. Tech Stack: PostgreSQL, Kubernetes, Backstage, Argo.",
                "url": f"https://www.{company_name.lower().replace(' ', '')}.com/about",
                "source_type": "company_website"
            },
            {
                "content": f"# {company_name} Core Engine\nThis repository contains the Go and Rust services for order routing.",
                "url": f"https://github.com/{company_name.lower().replace(' ', '')}/core",
                "source_type": "github_readme"
            }
        ]
    else:
        print("    [~] Phase A: Scraping (Web, GitHub, News, Community)...")
        queries = {
            f"{company_name} official website about product": "company_website",
            f"{company_name} github repository open source": "github_readme",
            f"{company_name} hiring engineer jobs careers": "jobs",
            f"{company_name} funding raised TechCrunch": "news",
            f"{company_name} site:news.ycombinator.com": "community",
            f"{company_name} site:reddit.com": "community",
            f"{company_name} engineering blog architecture": "community",
            f"{company_name} CEO CTO founder interview": "social"
        }
        
        for query, intent_type in queries.items():
            try:
                client = TavilyClient(api_key=TAVILY_API_KEYS.get_next_key())
                # Advanced search depth gets richer content, no truncation
                res = client.search(query=query, max_results=2, search_depth="advanced")
                for r in res.get('results', []):
                    # Classify URL
                    final_type = _classify_source(r.get('url', ''), intent_type)
                    scraped_data.append({
                        "content": r.get('content', ''),
                        "url": r.get('url', 'N/A'),
                        "source_type": final_type
                    })
            except Exception as e:
                print(f"        [!] Query failed ({query}): {e}")

    print(f"    [+] Scraped {len(scraped_data)} unique raw sources.")

    # ----------------------------------------------------
    # Phase B: INDEX
    # ----------------------------------------------------
    print("    [~] Phase B: Chunking and Indexing into Qdrant...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    
    texts_to_embed = []
    metadatas_to_embed = []
    
    for item in scraped_data:
        if not item["content"]:
            continue
        chunks = splitter.split_text(item["content"])
        for chunk in chunks:
            texts_to_embed.append(chunk)
            metadatas_to_embed.append({
                "company_name": company_name,
                "source_type": item["source_type"],
                "url": item["url"],
                "timestamp_epoch": int(time.time())
            })
            
    if texts_to_embed:
        print(f"    [~] Pushing {len(texts_to_embed)} chunks to Knowledge Store...")
        store = TreclKnowledgeStore()
        # Optionally clear old data for this company to avoid stale context
        store.clear(company_name)
        store.ingest(texts=texts_to_embed, metadatas=metadatas_to_embed)
        print("    [+] Indexing Complete.")
    else:
        print("    [!] Warning: No text chunks extracted. VectorDB will be empty.")

    # ----------------------------------------------------
    # Phase C: SYNTHESIZE
    # ----------------------------------------------------
    print("    [~] Phase C: Synthesizing core company summary from VectorDB...")
    summary = "Failed to extract summary."
    
    try:
        store = TreclKnowledgeStore()
        # Search for core company facts (no filter, general query)
        results = store.search(
            query=f"What does {company_name} do, what is their tech stack, funding, size, and technical challenges?",
            company_name=company_name,
            top_k=8
        )
        
        context_block = "\n\n".join([f"Source ({r['source_type']}): {r['content']}" for r in results])
        
        prompt = f"""You are a senior technical recruiter and engineering leader researching a startup.
        
Company Name: {company_name}

Here is the raw data retrieved from our Vector Database:
{context_block}

Synthesize a dense, bulleted summary covering exactly these 5 points if found (state 'Unknown' if not found):
- What they do (product/service)
- Tech stack (languages, frameworks, infra)
- Funding stage and amount
- Company size / scale signals
- One specific technical challenge they likely face based on their product.

Focus on facts. No fluff."""

        llm = _get_llm()
        response = llm.invoke(prompt)
        summary = response.content
        print("    [+] Synthesis Complete.")
    except Exception as e:
        print(f"    [!] Synthesis failed: {e}")

    print(f"[*] Data Ingestion Pipeline fully complete for {company_name}.")
    
    return {
        "company_summary": summary,
        "knowledge_store_ready": True
    }
