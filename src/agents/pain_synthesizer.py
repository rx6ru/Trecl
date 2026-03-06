"""
Pain Synthesizer Node implementation.
Responsible for reading company research and job signals, merging them with
the user's technical profile, and deducing actionable project ideas and pain points.

Priority hierarchy: Company Pain Point → Proposed Solution → User Stack Fit
"""

from pydantic import BaseModel, Field
from core.state import TreclState
from core.knowledge_store import TreclKnowledgeStore
from llm.model import llm
from langchain_core.messages import SystemMessage, HumanMessage

class SynthesizerOutput(BaseModel):
    """Structured JSON output schema for the Pain Synthesizer."""
    pain_points_ranked: str = Field(
        default="",
        description="A prioritized list of the company's top 3 technical pain points, formatted as markdown text."
    )
    project_ideas: str = Field(
        default="",
        description="A highly specific, custom project pitch that solves one of their pain points using the candidate's exact tech stack. Formatted as markdown text."
    )

def pain_synthesizer_node(state: TreclState) -> dict:
    """
    LangGraph node representing the Pain Synthesizer Agent.
    
    1. Reads research, user profile, and SELECTED TARGETS from the state.
    2. Queries VectorDB for deep technical context using TreclKnowledgeStore.
    3. Prompts the LLM to deduce pain points and propose a customized project.
    4. Outputs a validated Pydantic object mapping to `pain_points_ranked` and `project_ideas`.
    
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
    
    # [NEW] Query Knowledge Store for deep technical context
    deep_context = ""
    try:
        store = TreclKnowledgeStore()
        # Query specifically for engineering blogs and architectural pain points
        # limiting to 3 chunks to avoid blowing up the prompt
        results = store.search(
            query=f"What are the hardest engineering and scaling challenges at {company}? What is their core architecture?",
            company_name=company,
            top_k=3
        )
        if results:
            deep_context = "=== DEEP TECHNICAL CONTEXT (From VectorDB) ===\n"
            for r in results:
                deep_context += f"Source ({r.get('source_type', 'unknown')}): {r.get('content', '')}\n\n"
    except Exception as e:
        print(f"    [!] Failed to query Knowledge Store: {e}")
    
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

{deep_context}

Hiring Signals:
{jobs}

{targets_context}

=== CANDIDATE PROFILE ===
- Domain: {user_domain}
- Tech Stack: {user_stack}

=== YOUR TASK ===

STEP 1 — Identify Pain Points (HIGHEST PRIORITY):
Analyze ALL available data (research, jobs, deep context, selected targets) and deduce the company's 
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

=== OUTPUT FORMAT ===
You MUST return your output as a JSON object with EXACTLY two keys:
1. "pain_points_ranked" — a markdown-formatted string with the top 3 pain points.
2. "project_ideas" — a markdown-formatted string with the project pitch, scope, and stack-fit.
Both keys are REQUIRED. Do NOT merge them into one field.
"""

    messages = [
        SystemMessage(content="""You are an expert technical consultant. Prioritize the company's actual engineering needs above all else. Be genuine, not salesy.

CRITICAL: You MUST respond in perfectly valid JSON matching the exact schema requested. Do NOT return raw markdown outside of the JSON keys. 
Example response format:
{
  "pain_points_ranked": "## Top 3 Pain Points\\n1. First pain point...\\n2. Second pain point...",
  "project_ideas": "### One-Week Project\\n**Goal** - Directly relieve pain point #1..."
}"""),
        HumanMessage(content=synthesis_prompt)
    ]
    
    # Enforce structured output — try twice before falling back to raw
    structured_llm = llm.with_structured_output(SynthesizerOutput)
    
    for attempt in range(2):
        try:
            response = structured_llm.invoke(messages)
            
            # Validate both fields have content
            pain = response.pain_points_ranked or ""
            ideas = response.project_ideas or ""
            
            if pain and ideas:
                print("[+] Synthesis Complete.")
                return {
                    "pain_points_ranked": pain,
                    "project_ideas": ideas
                }
            elif pain and not ideas:
                # LLM merged everything into pain_points — split heuristically
                print(f"    [!] Attempt {attempt+1}: project_ideas empty, retrying...")
                continue
            else:
                print(f"    [!] Attempt {attempt+1}: incomplete output, retrying...")
                continue
                
        except Exception as e:
            print(f"    [!] Attempt {attempt+1} structured synthesis failed: {e}")
            continue
    
    # Final fallback: raw LLM output, split by a heuristic marker
    print("    [*] Falling back to raw LLM synthesis...")
    response = llm.invoke(messages)
    raw = response.content
    
    # Try to split on "Project" or "Proposal" headings
    split_markers = ["## 2", "## Project", "## Proposal", "STEP 2", "### Project"]
    for marker in split_markers:
        if marker in raw:
            idx = raw.index(marker)
            print("[+] Synthesis Complete (fallback, split).")
            return {
                "pain_points_ranked": raw[:idx].strip(),
                "project_ideas": raw[idx:].strip()
            }
    
    print("[+] Synthesis Complete (fallback, unsplit).")
    return {
        "pain_points_ranked": raw,
        "project_ideas": ""
    }
