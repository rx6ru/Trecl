"""
Pain Synthesizer Node implementation.
Responsible for reading company research and job signals, merging them with
the user's technical profile, and deducing actionable project ideas and pain points.
"""

import json
from core.state import TreclState
from llm.model import llm
from langchain_core.messages import SystemMessage, HumanMessage

def pain_synthesizer_node(state: TreclState) -> dict:
    """
    LangGraph node representing the Pain Synthesizer Agent.
    
    1. Reads research and user profile from the state.
    2. Prompts the LLM to deduce pain points and propose a customized project.
    3. Outputs JSON mapping to `pain_points_ranked` and `project_ideas`.
    
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
    
    Return EXACTLY the following JSON format (no markdown code blocks, just raw JSON):
    {{
      "pain_points_ranked": "1. [Point]\\n2. [Point]\\n3. [Point]",
      "project_ideas": "Project: [Name]\\nDetails: [How it solves the problem using the candidate's stack]"
    }}"""

    messages = [
        SystemMessage(content="You are a JSON-only API. You output raw valid JSON and nothing else."),
        HumanMessage(content=synthesis_prompt)
    ]
    
    response = llm.invoke(messages)
    
    try:
        # Clean the response just in case the LLM wrapped it in markdown
        cleaned = response.content.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.replace("```json", "", 1)
        if cleaned.endswith("```"):
            cleaned = cleaned[::-1].replace("```", "", 1)[::-1]
            
        data = json.loads(cleaned.strip())
        pain_points = data.get("pain_points_ranked", "Failed to parse pain points.")
        project = data.get("project_ideas", "Failed to parse project ideas.")
    except Exception as e:
        print(f"[!] Error parsing synthesizer JSON: {e}")
        pain_points = "Error analyzing pain points."
        project = "Error generating project idea."
    
    print("[+] Synthesis Complete.")
    return {
        "pain_points_ranked": pain_points,
        "project_ideas": project
    }
