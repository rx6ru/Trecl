"""
GitHub Analyst Sub-Graph.
A ReAct agent that autonomously navigates a company's GitHub organization
to find high-signal open source issues and PRs.
"""

from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
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

# --- Define the Built-in Tools ---

GITHUB_TOOLS = [
    list_org_repos,
    get_repo_stats,
    get_repo_labels,
    search_issues,
    search_prs,
    read_issue_thread
]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(GITHUB_TOOLS)


# --- ReAct Nodes ---

def github_reasoner(state: GithubAnalystState) -> dict:
    """
    The reasoning core of the GitHub Analyst sub-graph.
    """
    messages = state.get("messages", [])
    company_name = state.get("company_name", "Unknown Company")
    
    # Initialize with the system prompt on the first pass
    if not messages:
        # First, resolve the handle before prompting the agent
        print(f"\n[~] Starting GitHub Analyst for {company_name}...")
        handle = resolve_github_handle(company_name)
        
        if not handle:
            print(f"[!] Could not resolve GitHub handle for {company_name}.")
            # Give the agent a dummy handle or instruct it to fail gracefully
            handle = company_name.lower().replace(" ", "")
        else:
            print(f"    [*] Resolved official handle: {handle}")
            
        sys_msg = SystemMessage(content=f"""
        You are an expert Open Source Strategist analyzing the '{handle}' GitHub organization.
        Your goal is to find 3-5 high-impact open issues AND unreviewed PRs.

        You MUST follow this exact sequence. Skipping steps will cause tool errors.

        STEP 1: Call list_org_repos("{handle}") to discover real repositories.
                Do NOT guess or invent repository names.
        STEP 2: Call get_repo_stats on the top 2-3 repos to check health.
                Skip any repo where open_prs_count > open_issues_count (PR black hole).
        STEP 3: Call get_repo_labels on healthy repos to discover exact label names.
                Do NOT guess labels like "bug" or "P0" — they may not exist.
        STEP 4: Call search_issues using ONLY labels returned by get_repo_labels.
                Sort by "comments" to find the most painful issues.
        STEP 5: Call search_prs on the same repos to find unreviewed community PRs.
                Use search_prs, NOT search_issues, for Pull Requests.
        STEP 6: Call read_issue_thread on the top 2-3 issues to verify they are
                legitimate, unsolved, and technically substantive.

        IMPORTANT CONSTRAINTS:
        - Call only ONE tool at a time. Wait for its result before calling the next tool.
        - search_issues is for ISSUES only. search_prs is for PULL REQUESTS only.
        - Do NOT hallucinate repository names. Use ONLY repos from list_org_repos.
        - Do NOT guess label names. Use ONLY labels from get_repo_labels.
        - Keep limit=5 on search calls to conserve API rate limits.

        When done, summarize your final selections with title, URL, and repo_name.
        """)
        human_msg = HumanMessage(content=f"Find issues and PRs for {handle}")
        messages = [sys_msg, human_msg]
        
        # Invoke LLM with the fresh prompt
        response = llm_with_tools.invoke(messages)
        
        # CRITICAL: Return ALL three messages so add_messages persists the
        # system prompt and human message to state. Without this, the agent
        # forgets its instructions after the first tool call.
        return {"messages": [sys_msg, human_msg, response]}
    
    # Subsequent passes: state already has the full history, just invoke and append
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: GithubAnalystState) -> Literal["continue", "end"]:
    """
    Conditional edge: loop back to tools if requested, or exit to formatter.
    """
    last_message = state["messages"][-1]
    
    if last_message.tool_calls:
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
    Extracts the final structured `GithubIssue` and `GithubPR` lists from the agent's monologue.
    """
    extraction_prompt = SystemMessage(content="""
    Extract the finalized list of GitHub issues and PRs the agent gathered from the conversation history.
    Ensure they strictly match the Pydantic schema.
    """)
    
    structured_llm = llm.with_structured_output(GithubAnalystOutput)
    
    # We pass the full message history so the extractor sees what the agent decided
    response = structured_llm.invoke([extraction_prompt] + list(state["messages"]))
    
    print(f"    [*] Analyst extracted {len(response.github_issues)} issues and {len(response.github_prs)} PRs.")
    
    # Return the exact keys the main TreclState expects
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

# Compile with recursion_limit to prevent infinite ReAct loops
github_analyst_subgraph = builder.compile()
RECURSION_LIMIT = 25  # Max iterations before forced termination


# --- Wrapper Node for the Main Graph ---

def github_analyst_node(state: TreclState) -> dict:
    """
    The wrapper node that executes the compiled sub-graph.
    Takes TreclState string, initializes sub-graph state, and runs it.
    """
    # Reset guardrails from any previous run
    reset_guardrails()
    
    initial_subgraph_state = {
        "company_name": state["company_name"],
        "messages": []
    }
    
    try:
        # Execute the sub-graph with recursion limit
        final_state = github_analyst_subgraph.invoke(
            initial_subgraph_state,
            config={"recursion_limit": RECURSION_LIMIT}
        )
        
        # Pass the formatted output back up to the main graph
        return {
            "github_issues": final_state.get("github_issues", []),
            "github_prs": final_state.get("github_prs", [])
        }
    except Exception as e:
        print(f"[!] GitHub Analyst sub-graph failed: {e}")
        return {"github_issues": [], "github_prs": []}
