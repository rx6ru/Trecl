"""
Cold Email Writer Node implementation.
Generates context-aware, per-target outreach drafts based on the user's 
selected targets. Produces one draft per selected target plus a fused version
if multiple targets are selected.
"""

from core.state import TreclState
from llm.model import llm
from langchain_core.messages import SystemMessage, HumanMessage


# --- Tier-Specific Prompt Templates ---

TIER_1_INSTRUCTIONS = """Write an APPLICATION FOLLOW-UP email for THIS SPECIFIC ROLE: "{title}".
The user has applied (or is about to apply) for this role. The email should:
- Reference this specific role by its exact title
- Briefly mention a relevant project or skill that matches THIS role's JD
- Express genuine interest in the company's mission
- Ask for a brief chat or next steps
- Be under 100 words. Founders read short emails.
- Do NOT mention any other roles or opportunities."""

TIER_2_INSTRUCTIONS = """Write an OSS CONTRIBUTION email about THIS SPECIFIC item: "{title}".
The user has (or plans to) submit a PR fixing/improving this specific issue or PR. The email should:
- Reference this specific issue/PR by number and title
- Briefly explain what the technical fix involves in 1 sentence
- Express interest in contributing more, possibly as an intern/engineer
- DO NOT beg or oversell. The PR speaks for itself.
- Be under 100 words. Let the code do the talking.
- Do NOT mention any other issues, PRs, or roles."""

TIER_3_INSTRUCTIONS = """Write a COLD OUTREACH email. The user built a custom project that addresses 
a pain point the company has. The email should:
- Lead with the specific pain point you identified from research
- Describe the project in 1-2 sentences with a link
- Explain how it directly addresses their problem
- Offer to discuss or demo it
- Be under 120 words. No fluff, no flattery."""

FUSED_INSTRUCTIONS = """Write a COMBINED outreach email that weaves ALL of these actions into 
one cohesive message. This should feel natural, not like a bulleted list. The email should:
- Lead with the strongest signal (a merged PR > an application > a cold project)
- Naturally connect the different contributions/applications
- Sound like someone who genuinely cares about the company's engineering problems
- End with a clear call to action
- Be under 150 words."""


def cold_email_writer_node(state: TreclState) -> dict:
    """
    LangGraph node representing the Writer Agent.
    
    Generates one email draft per selected target, plus a fused draft
    if multiple targets are selected. Each draft is customized to the
    specific opportunity's context.
    
    Args:
        state (TreclState): The current graph state.
        
    Returns:
        dict: A dictionary update patch containing the `cold_email` key.
    """
    company = state["company_name"]
    pain_points = state.get("pain_points_ranked", "No pain points found.")
    project = state.get("project_ideas", "No project idea found.")
    selected_targets = state.get("selected_targets", [])
    
    print(f"\n[~] Writing targeted cold email(s) for {company}...")

    # Determine the right instructions per target based on its tier
    def get_instructions(target: dict) -> str:
        tier = target.get("action_tier", "")
        title = target.get("title", "Unknown")
        if "Tier 1" in tier:
            return TIER_1_INSTRUCTIONS.format(title=title)
        elif "Tier 2" in tier:
            return TIER_2_INSTRUCTIONS.format(title=title)
        else:
            return TIER_3_INSTRUCTIONS
    
    all_drafts = []
    
    # Generate one draft per selected target
    for target in selected_targets:
        title = target.get("title", "Untitled")
        tier = target.get("action_tier", "Unknown")
        url = target.get("url", "N/A")
        description = target.get("description", "")
        instructions = get_instructions(target)
        
        # Normalize tier label
        tier_label = tier.split(":")[0].strip() if ":" in tier else tier
        
        prompt = f"""Write an outreach email from a software developer to {company}.

Target Opportunity:
- Title: {title}
- URL: {url}
- Description: {description}

Company's Top Pain Points:
{pain_points}

My Proposed Project (use if relevant):
{project}

{instructions}

Format output strictly as:
SUBJECT: [subject line]

[email body]"""

        messages = [
            SystemMessage(content="You write concise, genuine outreach emails. No corporate buzzwords. No flattery. Direct and human."),
            HumanMessage(content=prompt)
        ]
        
        response = llm.invoke(messages)
        all_drafts.append(f"--- DRAFT: {title} ({tier_label}) ---\n{response.content}")
    
    # Generate a fused draft if multiple targets selected
    if len(selected_targets) > 1:
        all_targets_text = "\n".join([
            f"  - [{t.get('action_tier', '?')}] {t.get('title', 'Untitled')} ({t.get('url', 'N/A')})\n    {t.get('description', '')}"
            for t in selected_targets
        ])
        
        fused_prompt = f"""Write an outreach email from a software developer to {company}.

ALL Actions Taken:
{all_targets_text}

Company's Top Pain Points:
{pain_points}

My Proposed Project:
{project}

{FUSED_INSTRUCTIONS}

Format output strictly as:
SUBJECT: [subject line]

[email body]"""

        messages = [
            SystemMessage(content="You write concise, genuine outreach emails. No corporate buzzwords. No flattery. Direct and human."),
            HumanMessage(content=fused_prompt)
        ]
        
        response = llm.invoke(messages)
        all_drafts.append(f"--- DRAFT: COMBINED (All Selected) ---\n{response.content}")
    
    combined_output = "\n\n".join(all_drafts)
    
    print(f"\n[+] Draft Complete. Generated {len(all_drafts)} email draft(s).")
    return {"cold_email": combined_output}
