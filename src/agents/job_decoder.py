"""
Job Decoder Node implementation.
Responsible for scraping the web using Tavily specifically for hiring signals,
open roles, and engineering team pain points, then synthesizing them.
"""

from core.state import TreclState
from tools.search import perform_job_research
from llm.model import llm
from langchain_core.messages import SystemMessage, HumanMessage

def job_decoder_node(state: TreclState) -> dict:
    """
    LangGraph node representing the Job Decoder Agent.
    
    1. Extracts the target company name.
    2. Performs targeted web searches for open engineering roles.
    3. Prompts the LLM to synthesize the raw hiring data into key structural needs.
    
    Args:
        state (TreclState): The current graph state.
        
    Returns:
        dict: A dictionary update patch containing the `company_jobs` key.
    """
    company = state["company_name"]
    print(f"\n[*] Decoding hiring signals for {company}...")

    # Step 1: Gather raw internet job data
    raw_jobs = perform_job_research(company)
    
    # Step 2: Synthesis prompt
    synthesis_prompt = f"""Based on this web research about {company}'s hiring and careers, extract their core needs:
    
    1. Open Engineering Roles (list specific titles mentioned, e.g. "Senior Backend Engineer")
    2. Core Tech Stack Requirements for these roles
    3. Inferred Pain Points (e.g. if hiring 5 DevOps engineers, they likely have infrastructure scaling pain)
    
    Web research:
    {raw_jobs}
    
    Be concise, specific, and fact-based."""

    messages = [
        SystemMessage(content="You are an expert technical recruiter analyzing startup engineering needs. Output clean, fact-checked bullet points."),
        HumanMessage(content=synthesis_prompt)
    ]
    
    # Step 3: Extract & Synthesize
    response = llm.invoke(messages)
    
    print("\n[+] Job Decoding Complete.")
    return {"company_jobs": response.content}
