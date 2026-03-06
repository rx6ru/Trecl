"""
GitHub Analyst Sub-Graph.
A ReAct agent that autonomously navigates a company's GitHub organization
to find high-signal open source issues and PRs.
"""

from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from core.state import TreclState, GithubAnalystState, GithubIssue, GithubPR
from llm.model import llm
from tools.github import (
    resolve_github_handle,
    reset_guardrails,
    list_org_repos,
    get_repo_stats,
    get_repo_labels,
    search_issues,
    search_prs,
    read_issue_thread
)
from tools.knowledge import search_company_knowledge

# --- Define the Built-in Tools ---

GITHUB_TOOLS = [
    search_company_knowledge,
    list_org_repos,
    get_repo_stats,
    get_repo_labels,
    search_issues,
    search_prs,
    read_issue_thread
]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(GITHUB_TOOLS)

# Maximum number of reasoner invocations before we force a graceful exit.
# Each reasoner→tools round-trip = 1 invocation. With a 7-step protocol
# plus retries on label rejections, 15 gives plenty of headroom.
MAX_REASONER_CALLS = 15
RECURSION_LIMIT = 40  # Hard ceiling (2 graph steps per call → 40 covers MAX_REASONER_CALLS)


# --- ReAct Nodes ---

def github_reasoner(state: GithubAnalystState) -> dict:
    """
    The reasoning core of the GitHub Analyst sub-graph.
    Includes iteration tracking to force graceful exit before recursion limit.
    """
    messages = state.get("messages", [])
    company_name = state.get("company_name", "Unknown Company")
    
    # Count how many times the reasoner has been invoked by counting AI messages
    ai_message_count = sum(1 for m in messages if isinstance(m, AIMessage))
    
    # If we're approaching the limit, force the agent to stop calling tools
    if ai_message_count >= MAX_REASONER_CALLS:
        print(f"    [!] GitHub Analyst reached {MAX_REASONER_CALLS} iterations — forcing summary.")
        stop_msg = AIMessage(content=(
            "I have reached my iteration budget. Based on the data I've gathered so far, "
            "here is my summary of findings for this organization. "
            "I will now output my final selections."
        ))
        return {"messages": [stop_msg]}
    
    # Initialize with the system prompt on the first pass
    if not messages:
        print(f"\n[~] Starting GitHub Analyst for {company_name}...")
        handle = resolve_github_handle(company_name)
        
        if not handle:
            print(f"[!] Could not resolve GitHub handle for {company_name}.")
            handle = company_name.lower().replace(" ", "")
        else:
            print(f"    [*] Resolved official handle: {handle}")
            
        sys_msg = SystemMessage(content=f"""
        You are an expert Open Source Strategist analyzing the '{handle}' GitHub organization.
        Your goal is to find 3-5 high-impact open issues AND unreviewed PRs.

        You MUST follow this sequence. Skipping steps will cause tool errors.

        STEP 1: Call search_company_knowledge(query="github repositories", company_name="{company_name}", source_filter="github_readme") 
                to check if VectorDB has repository data from web scraping.
        STEP 2: Call list_org_repos("{handle}") to discover real repositories.
        STEP 3: Call get_repo_stats on the TOP 1-2 repos ONLY (by star count).
                Skip any repo where open_prs_count > open_issues_count.
        STEP 4: Call get_repo_labels on ONE healthy repo to discover exact label names.
                Do NOT guess labels — they may not exist.
        STEP 5: Call search_issues using ONLY labels returned by get_repo_labels.
                If no useful labels exist, call search_issues WITHOUT labels.
                Sort by "comments". Keep limit=5.
        STEP 6: Call search_prs on the same repo. Keep limit=3.
        STEP 7: STOP. Summarize your final selections with title, URL, and repo_name.
                Do NOT call read_issue_thread unless you have fewer than 2 issues.

        CRITICAL CONSTRAINTS:
        - You have a STRICT budget of {MAX_REASONER_CALLS} tool calls. Be efficient.
        - If a tool returns a REJECTED error, do NOT retry with the same invalid arguments.
          Instead, adjust your approach (e.g., drop the label filter entirely).
        - Do NOT hallucinate repository or label names.
        - When you have enough data, STOP and summarize. Do not keep exploring.
        """)
        human_msg = HumanMessage(content=f"Find issues and PRs for {handle} (Full Company Name: {company_name})")
        messages = [sys_msg, human_msg]
        
        response = llm_with_tools.invoke(messages)
        return {"messages": [sys_msg, human_msg, response]}
    
    # Subsequent passes
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: GithubAnalystState) -> Literal["continue", "end"]:
    """
    Conditional edge: loop back to tools if requested, or exit to formatter.
    """
    last_message = state["messages"][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "continue"
    return "end"


# --- Formatter Node (Pydantic Extraction) ---

class GithubAnalystOutput(BaseModel):
    """Structured output expected by the main TreclState."""
    github_issues: list[dict] = Field(
        description="List of selected GitHub issues. Each dict must have 'title', 'url', and 'repo_name'."
    )
    github_prs: list[dict] = Field(
        description="List of selected GitHub Pull Requests. Each dict must have 'title', 'url', and 'repo_name'."
    )


def format_output(state: GithubAnalystState) -> dict:
    """
    Extracts the final structured GithubIssue and GithubPR lists from the agent's monologue.
    """
    extraction_prompt = SystemMessage(content="""
    Extract the finalized list of GitHub issues and PRs the agent gathered from the conversation history.
    Ensure they strictly match the Pydantic schema. If the agent found no issues or PRs, return empty lists.
    """)
    
    structured_llm = llm.with_structured_output(GithubAnalystOutput)
    
    response = structured_llm.invoke([extraction_prompt] + list(state["messages"]))
    
    print(f"    [*] Analyst extracted {len(response.github_issues)} issues and {len(response.github_prs)} PRs.")
    
    return {
        "github_issues": response.github_issues,
        "github_prs": response.github_prs
    }


# --- Build the Sub-Graph ---

builder = StateGraph(GithubAnalystState)

# Nodes
builder.add_node("github_reasoner", github_reasoner)
builder.add_node("github_tools", ToolNode(GITHUB_TOOLS))
builder.add_node("format_output", format_output)

# Edges
builder.set_entry_point("github_reasoner")
builder.add_conditional_edges(
    "github_reasoner",
    should_continue,
    {
        "continue": "github_tools",
        "end": "format_output"
    }
)
builder.add_edge("github_tools", "github_reasoner")
builder.add_edge("format_output", END)

github_analyst_subgraph = builder.compile()


# --- Wrapper Node for the Main Graph ---

def github_analyst_node(state: TreclState) -> dict:
    """
    The wrapper node that executes the compiled sub-graph.
    Takes TreclState, initializes sub-graph state, and runs it.
    """
    reset_guardrails()
    
    initial_subgraph_state = {
        "company_name": state["company_name"],
        "messages": []
    }
    
    try:
        final_state = github_analyst_subgraph.invoke(
            initial_subgraph_state,
            config={"recursion_limit": RECURSION_LIMIT}
        )
        
        return {
            "github_issues": final_state.get("github_issues", []),
            "github_prs": final_state.get("github_prs", [])
        }
    except Exception as e:
        print(f"[!] GitHub Analyst sub-graph failed: {e}")
        return {"github_issues": [], "github_prs": []}
