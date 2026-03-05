"""
Pain Synthesizer Node implementation.
Responsible for reading company research and job signals, merging them with
the user's technical profile, and deducing actionable project ideas and pain points.

Priority hierarchy: Company Pain Point → Proposed Solution → User Stack Fit
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
    
    1. Reads research, user profile, and SELECTED TARGETS from the state.
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
    
    # Read from selected_targets (HITL output) instead of raw issues/PRs
    selected_targets = state.get("selected_targets", [])
    
    print(f"\n[*] Synthesizing pain points & project ideas for {company}...")
    
    # Format the selected targets for context
    targets_context = "No specific targets selected."
    if selected_targets:
        targets_context = "User's Selected Targets:\n"
        for t in selected_targets:
            targets_context += f"  - [{t.get('action_tier', 'Unknown')}] {t.get('title', 'Untitled')}\n"
            targets_context += f"    Type: {t.get('type', 'N/A')} | URL: {t.get('url', 'N/A')}\n"
            targets_context += f"    Relevance: {t.get('relevance', 'N/A')}\n"

    synthesis_prompt = f"""You are an elite technical consultant. Your job is to help a candidate 
genuinely solve a company's engineering problems — not to write a sales pitch.

=== COMPANY CONTEXT ===
Company: {company}

Research:
{research}

Hiring Signals:
{jobs}

{targets_context}

=== CANDIDATE PROFILE ===
- Domain: {user_domain}
- Tech Stack: {user_stack}

=== YOUR TASK ===

STEP 1 — Identify Pain Points (HIGHEST PRIORITY):
Analyze ALL available data (research, jobs, selected targets) and deduce the company's 
top 3 technical pain points. These must be genuine engineering bottlenecks — not generic 
observations. Look at:
  - What roles they're hiring for (many DevOps hires = infra pain)
  - What issues/PRs are open in their repos (stale = bottleneck)
  - What their product does vs what's hard about scaling it

STEP 2 — Propose a Project (COMPANY-FIRST):
Pitch exactly ONE specific, highly technical project that:
  a) Directly addresses their #1 pain point (this is NON-NEGOTIABLE)
  b) Would genuinely help them if built, not just demonstrate the candidate's skills
  c) Is scoped to 1-2 weeks of work (realistic for an intern to deliver)
  d) Sounds like something the candidate actually wants to build — genuine curiosity, not flattery

STEP 3 — Connect to User's Stack (SUPPORTING, NOT LEADING):
Explain how the candidate's stack ({user_stack}) enables them to execute this project.
If the stack isn't a perfect fit, focus on: transferable engineering skills, relevant 
problem-solving experience, and willingness to learn their specific tools.

IMPORTANT: The project must convince the company that hiring this person solves a real 
problem they have TODAY. If the suggestion sounds like a school project or a generic 
portfolio piece, you have FAILED.

If there are relevant open-source targets (issues, PRs), you may alternatively suggest 
solving a specific issue or contributing to a stale PR as your pitched 'project'.
"""

    messages = [
        SystemMessage(content="You are an expert technical consultant. Prioritize the company's actual engineering needs above all else. Be genuine, not salesy."),
        HumanMessage(content=synthesis_prompt)
    ]
    
    # Enforce Pydantic structured output directly at the Langchain LLM level
    try:
        structured_llm = llm.with_structured_output(SynthesizerOutput)
        response = structured_llm.invoke(messages)
        
        print("[+] Synthesis Complete.")
        return {
            "pain_points_ranked": response.pain_points_ranked,
            "project_ideas": response.project_ideas
        }
    except Exception as e:
        print(f"    [!] Structured synthesis failed: {e}")
        print("    [*] Falling back to raw LLM synthesis...")
        
        response = llm.invoke(messages)
        print("[+] Synthesis Complete (fallback).")
        return {
            "pain_points_ranked": response.content,
            "project_ideas": ""
        }
