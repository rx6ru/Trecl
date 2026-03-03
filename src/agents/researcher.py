"""
Company Researcher Node implementation.
Responsible for scraping the web using Tavily and processing raw data using Cerebras
to build a structured intelligence profile of a target company.
"""

from core.state import TreclState
from tools.search import perform_deep_company_research
from llm.model import llm
from langchain_core.messages import SystemMessage, HumanMessage

def company_researcher_node(state: TreclState) -> dict:
    """
    LangGraph node representing the Research Agent.
    
    1. Extracts the target company name from the current state.
    2. Performs deep parallel web searches using the Tavily client.
    3. Prompts the LLM to synthesize the raw web data into 5 specific factual bullets.
    
    Args:
        state (TreclState): The current graph state containing `company_name`.
        
    Returns:
        dict: A dictionary update patch containing the `company_summary` key.
    """
    company = state["company_name"]
    print(f"\n🕵️‍♂️ Researching {company} using Web Search...")

    # Step 1: Gather raw internet data
    raw_research = perform_deep_company_research(company)
    
    # Step 2: Define specific extraction instructions for v0.2
    synthesis_prompt = f"""Based on this web research about {company}, extract the following 5 points strictly:
    1. What they do (product/service)
    2. Tech stack (confirmed from sources)
    3. Funding stage and amount
    4. Company size/scale signals
    5. One specific technical challenge they likely face
    
    Web research:
    {raw_research}
    
    Be specific. Only include facts supported by the sources. Do not hallucinate."""

    messages = [
        SystemMessage(content="You are an expert technical business analyst and researcher. Output clean, fact-checked bullet points mapping strictly to the user's requested 5 points."),
        HumanMessage(content=synthesis_prompt)
    ]
    
    # Step 3: Extract & Synthesize
    response = llm.invoke(messages)
    
    print("\n✅ Research Complete.")
    return {"company_summary": response.content}
