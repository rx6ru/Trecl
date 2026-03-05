"""
Job Decoder Node implementation.
Responsible for scraping the web using Tavily specifically for hiring signals,
open roles, and engineering team pain points, then synthesizing them into
structured output with URLs and JD summaries.
"""

from pydantic import BaseModel, Field
from core.state import TreclState
from tools.search import perform_job_research
from llm.model import llm
from langchain_core.messages import SystemMessage, HumanMessage


# --- Pydantic Models for Structured Output ---

class JobListing(BaseModel):
    """A single job listing extracted from web research."""
    title: str = Field(description="The exact role title, e.g. 'Software Engineering Intern'")
    url: str = Field(description="The direct application URL. Use 'N/A' only if truly unavailable.")
    summary: str = Field(description="A 1-2 sentence summary of the role's responsibilities and requirements.")

class JobDecoderOutput(BaseModel):
    """Structured output from the Job Decoder agent."""
    job_overview: str = Field(
        description="A concise synthesis of the company's overall hiring needs, tech stack requirements, and inferred engineering pain points."
    )
    listings: list[JobListing] = Field(
        description="A list of specific, confirmed open roles with their URLs and summaries. May be empty if no listings are found."
    )


def job_decoder_node(state: TreclState) -> dict:
    """
    LangGraph node representing the Job Decoder Agent.
    
    1. Extracts the target company name.
    2. Performs targeted web searches for open engineering roles.
    3. Prompts the LLM to synthesize the raw hiring data into structured output.
    
    Args:
        state (TreclState): The current graph state.
        
    Returns:
        dict: A dictionary update patch containing the `company_jobs` key.
    """
    company = state["company_name"]
    print(f"\n[*] Decoding hiring signals for {company}...")

    # Step 1: Gather raw internet job data (now includes URLs from Tavily)
    raw_jobs = perform_job_research(company)
    
    # Step 2: Synthesis prompt
    synthesis_prompt = f"""Based on this web research about {company}'s hiring and careers, extract:
    
    1. job_overview: A synthesis of their overall hiring needs including:
       - Open Engineering Roles (list specific titles)
       - Core Tech Stack Requirements for these roles
       - Inferred Pain Points (e.g. if hiring 5 DevOps engineers, they likely have infrastructure scaling pain)
    
    2. listings: For each CONFIRMED open role found in the research, extract:
       - title: The exact job title
       - url: The direct link to the job posting (look for URLs from career pages, YC Work at a Startup, Wellfound, etc.)
       - summary: A 1-2 sentence description of what the role involves

    IMPORTANT: Extract actual URLs from the research data. Every listing in the research has a URL in parentheses.
    If a role is mentioned but has no direct link, use the company's careers page URL if available.
    
    Web research:
    {raw_jobs}
    
    Be concise, specific, and fact-based."""

    messages = [
        SystemMessage(content="You are an expert technical recruiter analyzing startup engineering needs. Extract structured data precisely."),
        HumanMessage(content=synthesis_prompt)
    ]
    
    # Step 3: Extract with structured output + fallback
    import json
    
    try:
        structured_llm = llm.with_structured_output(JobDecoderOutput)
        response = structured_llm.invoke(messages)
        
        # Build the combined text output for downstream nodes
        jobs_text = response.job_overview
        if response.listings:
            jobs_text += "\n\n**Confirmed Listings:**\n"
            for listing in response.listings:
                jobs_text += f"- {listing.title} → {listing.url}\n  {listing.summary}\n"
        
        print(f"\n[+] Job Decoding Complete. Found {len(response.listings)} confirmed listings.")
        return {"company_jobs": jobs_text}
        
    except Exception as e:
        print(f"    [!] Structured job decoding failed: {e}")
        print("    [*] Falling back to raw LLM synthesis...")
        
        # Fallback: plain text synthesis
        response = llm.invoke(messages)
        print("\n[+] Job Decoding Complete (fallback).")
        return {"company_jobs": response.content}
