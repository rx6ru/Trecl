import sys
import logging

logging.basicConfig(level=logging.INFO)

try:
    from src.core.knowledge_store import TreclKnowledgeStore
    
    print("Testing TreclKnowledgeStore logic...")
    store = TreclKnowledgeStore()
    
    company = "TestCorp_123"
    
    print("1. Testing Ingest...")
    store.ingest(
        texts=["TraceRoot is building the future of Agentic Security.", "They are hiring Go engineers to scale their data pipeline."],
        metadatas=[
            {"company_name": company, "source_type": "website", "url": "https://test.com/1", "timestamp_epoch": 1700000000},
            {"company_name": company, "source_type": "jobs", "url": "https://test.com/2", "timestamp_epoch": 1700000010}
        ]
    )
    print("   Ingest OK.")
    
    print("2. Testing Search...")
    results = store.search(query="Security and Go", company_name=company, top_k=2)
    print(f"   Found {len(results)} results:")
    for r in results:
        print(f"   - {r['content']} (Source: {r['source_type']})")
        
    print("3. Testing Clear...")
    store.clear(company)
    print("   Clear OK.")
    
    print("\nSUCCESS: Knowledge Store is wired correctly.")
except Exception as e:
    print(f"\nFAILED: {e}")
    sys.exit(1)
