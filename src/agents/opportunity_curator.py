"""
Opportunity Curator Node implementation.
Responsible for aggregating all research signals (jobs, GitHub issues, PRs),
aggressively filtering them against the user's anti-persona, assigning action tiers,
and producing a curated, ranked list of opportunities for Human-in-the-Loop selection.
"""

from pydantic import BaseModel, Field
from core.state import TreclState
from llm.model import llm
from langchain_core.messages import SystemMessage, HumanMessage


# --- Pydantic Models for Structured Output ---

class OpportunityItemModel(BaseModel):
    """Pydantic model matching the OpportunityItem TypedDict for structured output."""
    type: str = Field(
        description="Must be one of: 'job_posting', 'github_issue', 'github_pr', 'hackathon'"
    )
    title: str = Field(
        description="A clean, human-readable label for the opportunity."
    )
    description: str = Field(
        description="A 1-2 sentence summary of what the role involves or what the issue/PR is about. For jobs, summarize the JD. For issues, summarize the problem."
    )
    url: str = Field(
        description="The direct link to the job listing, GitHub issue, or PR."
    )
    source: str = Field(
        description="The agent that found this: 'job_decoder' or 'github_analyst'"
    )
    relevance: str = Field(
        description="A one-line explanation of why this matches the user's stack and domain."
    )
    action_tier: str = Field(
        description="Must be exactly one of: 'Tier 1: Active Listing', 'Tier 2: OSS Pitch', or 'Tier 3: Cold Outreach'"
    )
    suggested_action: str = Field(
        description="A concrete, strategic next step for the user to take (e.g., 'Submit PR fixing issue #142')."
    )


class CuratorOutput(BaseModel):
    """The final structured output from the opportunity_curator node."""
    curated_opportunities: list[OpportunityItemModel] = Field(
        description="A curated, aggressively filtered, and ranked list of the best opportunities. May be empty if all are filtered out."
    )


# --- The Curator Node ---

def opportunity_curator_node(state: TreclState) -> dict:
    """
    LangGraph node representing the Opportunity Curator Agent.
    
    This node sits AFTER the 3-way fan-in (researcher, job_decoder, github_analyst)
    and BEFORE the graph suspends for Human-in-the-Loop selection.
    
    Flow:
        1. Reads all raw signals: company_jobs, github_issues, github_prs.
        2. Reads the user's profile: user_stack, user_domain, user_anti_persona.
        3. Applies the anti-persona as a HARD EXCLUSION GATE on all opportunities.
        4. Classifies surviving opportunities into action tiers.
        5. Ranks by relevance to the user's stack.
        6. Returns a curated list of OpportunityItem dicts.
    
    Args:
        state (TreclState): The current graph state.
        
    Returns:
        dict: A dictionary update patch containing `curated_opportunities`.
    """
    company = state["company_name"]
    jobs = state.get("company_jobs", "No job data available.")
    issues = state.get("github_issues", [])
    prs = state.get("github_prs", [])
    user_stack = ", ".join(state.get("user_stack", []))
    user_domain = state.get("user_domain", "software engineer")
    anti_persona = state.get("user_anti_persona", "None specified.")
    summary = state.get("company_summary", "No company research available.")
    
    print(f"\n[*] Curating opportunities for {company}...")

    # --- Format raw signals into human-readable context ---
    
    jobs_context = f"Job Listings & Hiring Signals:\n{jobs}"
    
    oss_context = "Open Source Data: No accessible open source footprint found."
    if issues or prs:
        oss_context = "Open Source Data:\n"
        if issues:
            oss_context += "GitHub Issues:\n" + "\n".join(
                [f"  - [{i['repo_name']}] {i['title']} → {i['url']}" for i in issues]
            ) + "\n"
        if prs:
            oss_context += "GitHub PRs:\n" + "\n".join(
                [f"  - [{p['repo_name']}] {p['title']} → {p['url']}" for p in prs]
            ) + "\n"

    # --- Build the Curator Prompt ---
    
    curator_prompt = f"""You are an elite career intelligence agent curating opportunities for a targeted outreach campaign.

=== COMPANY CONTEXT ===
Company: {company}
{summary}

=== RAW SIGNALS (Your input data) ===

{jobs_context}

{oss_context}

=== CANDIDATE PROFILE ===
- Domain: {user_domain}
- Tech Stack: {user_stack}

=== STRICT EXCLUSION RULES (APPLY BEFORE ANYTHING ELSE) ===

The user has defined the following Anti-Persona — roles, tasks, and
responsibilities they CANNOT and DO NOT want to do:

Anti-Persona: {anti_persona}

HARD RULE — Apply this two-step mental check to EVERY opportunity:

  Step 1 — Job listings:
    Does the role's PRIMARY responsibility match the anti-persona?
    → YES: IMMEDIATELY DISCARD. Do not include it in the output.
    → NO: Proceed to evaluation.
    
    Does the role TITLE contain keywords from the anti-persona
    (e.g., "ML Researcher", "Data Scientist", "CUDA Engineer")?
    → YES: DISCARD *unless* the actual DESCRIPTION proves the work is applied/integration-focused.
    → NO: Proceed.

  Step 2 — GitHub Issues & PRs:
    Does the issue/PR require skills from the anti-persona to solve?
    (e.g., "Optimize PyTorch tensor memory allocation" requires ML/CUDA skills)
    → YES: DISCARD. An open-source issue requiring excluded skills is NOT an opportunity.
    → NO: Proceed.
    
    Is the issue/PR about infrastructure, integrations, tooling, APIs, or developer experience
    that the user CAN do with their stack ({user_stack})?
    → YES: KEEP. This is a valid opportunity.
    → NO: DISCARD.

  Example:
    Anti-persona: "No ML research, no model training, no CUDA"
    ✗ "Senior ML Research Engineer — fine-tune LLMs" → DISCARD (primary work is ML training)
    ✓ "ML Engineer Intern — build RAG pipelines and agent workflows" → KEEP (work is applied)
    ✗ GitHub Issue: "Optimize CUDA kernel for attention mechanism" → DISCARD (requires CUDA)
    ✓ GitHub Issue: "Add retry logic to OAuth token refresh" → KEEP (backend/integration work)

The output list must contain ZERO items that violate the anti-persona.
If aggressive filtering removes all opportunities, return an EMPTY list.
DO NOT pad the list with irrelevant entries to fill a quota.

=== TIER CLASSIFICATION (Apply to surviving opportunities) ===

Assign exactly one tier to each opportunity:

- "Tier 1: Active Listing" → The company has a confirmed open role the user can apply to NOW.
- "Tier 2: OSS Pitch" → No direct listing, but the company has open-source repos. The user should contribute (fix an issue, submit a PR) and THEN pitch themselves.
- "Tier 3: Cold Outreach" → No listing, no actionable OSS. The user should build a custom project solving a pain point and cold-email the founder.

=== RANKING ===

Sort the final list by:
1. Tier (Tier 1 first, then 2, then 3)
2. Within each tier, rank by relevance to the user's tech stack (highest stack overlap first)

=== OUTPUT REQUIREMENTS ===

For each opportunity, provide:
- type: "job_posting", "github_issue", or "github_pr"
- title: A clean label (e.g., "Backend AI Engineer Intern" or "Fix OAuth retry logic (#142)")
- description: A 1-2 sentence summary of what the role involves (for jobs: summarize the JD, responsibilities, and requirements; for issues/PRs: summarize the technical problem)
- url: The direct link. If no URL is available, use "N/A".
- source: "job_decoder" for job signals, "github_analyst" for issues/PRs
- relevance: One line on why this matches the user
- action_tier: Exactly one of the three tiers above
- suggested_action: A concrete next step (e.g., "Apply on YC Work at a Startup", "Submit PR fixing issue #142", "Build a FastAPI middleware demo and email the CTO")
"""

    messages = [
        SystemMessage(content="You are an opportunity curation specialist. Apply all exclusion rules strictly. Return structured output only."),
        HumanMessage(content=curator_prompt)
    ]
    
    # Attempt structured output with retry and JSON fallback
    import json
    
    curated = []
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            structured_llm = llm.with_structured_output(CuratorOutput)
            response = structured_llm.invoke(messages)
            curated = [item.model_dump() for item in response.curated_opportunities]
            break  # Success, exit retry loop
        except Exception as e:
            print(f"    [!] Structured output attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                print("    [*] Retrying...")
                continue
            
            # Final fallback: raw invoke + manual JSON parsing
            print("    [*] Falling back to raw LLM invoke + manual JSON parsing...")
            try:
                fallback_prompt = f"""Based on this data, return ONLY a valid JSON array of opportunity objects.
Each object must have these exact keys: "type", "title", "url", "source", "relevance", "action_tier", "suggested_action".
Do NOT include any text before or after the JSON array. Start with [ and end with ].

{curator_prompt}"""
                
                raw_response = llm.invoke([
                    SystemMessage(content="You return only valid JSON arrays. No explanations, no markdown, no code fences."),
                    HumanMessage(content=fallback_prompt)
                ])
                
                raw_text = raw_response.content.strip()
                # Strip markdown code fences if present
                if raw_text.startswith("```"):
                    raw_text = raw_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                
                parsed = json.loads(raw_text)
                if isinstance(parsed, list):
                    curated = parsed
                elif isinstance(parsed, dict) and "curated_opportunities" in parsed:
                    curated = parsed["curated_opportunities"]
                    
                print(f"    [+] Fallback parsing recovered {len(curated)} opportunities.")
            except Exception as fallback_err:
                print(f"    [!] Fallback also failed: {fallback_err}")
                print("    [!] Returning empty opportunities list.")
                curated = []
    
    print(f"[+] Curation Complete: {len(curated)} opportunities survived filtering.")
    for i, opp in enumerate(curated):
        print(f"    [{i+1}] [{opp.get('action_tier', 'Unknown')}] {opp.get('title', 'Untitled')}")
    
    return {
        "curated_opportunities": curated
    }
