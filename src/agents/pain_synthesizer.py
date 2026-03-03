"""
Pain Synthesizer Node implementation.
Responsible for reading company research and job signals, merging them with
the user's technical profile, and deducing actionable project ideas and pain points.
"""

from pydantic import BaseModel, Field
from core.state import TreclState
from llm.model import llm
from langchain_core.messages import SystemMessage, HumanMessage

class SynthesizerOutput(BaseModel):
    """Structured JSON output schema for the Pain Synthesizer."""
    pain_points_ranked: str = Field(
        description="A prioritized list of the company's top 3 technical pain points, formatted as text."
    )
    project_ideas: str = Field(
        description="A highly specific, custom project pitch that solves one of their pain points using the candidate's exact tech stack."
    )

def pain_synthesizer_node(state: TreclState) -> dict:
    """
    LangGraph node representing the Pain Synthesizer Agent.
    
    1. Reads research and user profile from the state.
    2. Prompts the LLM to deduce pain points and propose a customized project.
    3. Outputs a validated Pydantic object mapping to `pain_points_ranked` and `project_ideas`.
    
    Args:
        state (TreclState): The current graph state.
        
    Returns:
        dict: A dictionary update patch containing `pain_points_ranked` and `project_ideas`.
    """
    company = state["company_name"]
    research = state["company_summary"]
    jobs = state.get("company_jobs", "No specific job signals.")
    user_stack = ", ".join(state.get("user_stack", []))
    user_domain = state.get("user_domain", "software engineer")
    
    print(f"\n[*] Synthesizing pain points & project ideas for {company}...")

    synthesis_prompt = f"""You are an elite technical consultant.
    
    Company Research:
    {research}
    
    Hiring Needs & Signals:
    {jobs}
    
    Candidate Profile:
    - Domain: {user_domain}
    - Tech Stack: {user_stack}
    
    Based on the above, deduce the company's top 3 technical pain points.
    Then, pitch exactly ONE specific, highly technical project idea that this candidate could build to solve one of those pain points using their exact tech stack.
    """

    messages = [
        SystemMessage(content="You are an expert technical consultant. Always format outputs precisely as requested."),
        HumanMessage(content=synthesis_prompt)
    ]
    
    # Enforce Pydantic structured output directly at the Langchain LLM level
    structured_llm = llm.with_structured_output(SynthesizerOutput)
    response = structured_llm.invoke(messages)
    
    print("[+] Synthesis Complete.")
    return {
        "pain_points_ranked": response.pain_points_ranked,
        "project_ideas": response.project_ideas
    }
