"""
Trecl Main Entrypoint.
Orchestrates the multi-agent system using LangGraph.
"""

from langgraph.graph import StateGraph, START, END

from core.state import TreclState
from agents import (
    data_ingester_node,
    job_decoder_node,
    pain_synthesizer_node,
    github_analyst_node,
    cold_email_writer_node,
    opportunity_curator_node
)

from langgraph.checkpoint.sqlite import SqliteSaver

def build_graph(checkpointer=None):
    """
    Constructs the LangGraph execution flow.
    
    Nodes:
        - data_ingester: Scrapes the web, chunks and stores in VectorDB, then synthesizes facts.
        - job_decoder: Analyzes job postings.
        - github_analyst: Finds open source issues and PRs.
        - opportunity_curator: Filters, tiers, and ranks opportunities.
        - pain_synthesizer: Selects target and drafts a custom project pitch.
        - writer: Drafts a cold email using those facts.
        
    Flow:
                                     ---\ 
        START -> data_ingester  -->  ---> opportunity_curator -> pain_synthesizer -> writer -> END
                                     ---/
        
    Returns:
        CompiledGraph: The executable state machine.
    """
    # 1. Initialize Graph with central State
    graph = StateGraph(TreclState)
    
    # 2. Add Nodes (Agents)
    graph.add_node("data_ingester", data_ingester_node)
    graph.add_node("job_decoder", job_decoder_node)
    graph.add_node("github_analyst", github_analyst_node)
    graph.add_node("opportunity_curator", opportunity_curator_node)
    graph.add_node("pain_synthesizer", pain_synthesizer_node)
    graph.add_node("writer", cold_email_writer_node)
    
    # 3. Define the DAG Execution Order
    # Data Ingester blocks the fan-out to ensure VectorDB is populated
    graph.add_edge(START, "data_ingester")
    graph.add_edge("data_ingester", "job_decoder")
    graph.add_edge("data_ingester", "github_analyst")
    
    # Fan-in from discovery nodes to the curator
    graph.add_edge("job_decoder", "opportunity_curator")
    graph.add_edge("github_analyst", "opportunity_curator")
    
    graph.add_edge("opportunity_curator", "pain_synthesizer")
    graph.add_edge("pain_synthesizer", "writer")
    graph.add_edge("writer", END)
    
    # 4. Compile into an executable app
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["pain_synthesizer"] if checkpointer else None
    )


if __name__ == "__main__":
    print("=" * 60)
    print("TRECL INTELLIGENCE AGENT (v0.3 - HITL)")
    print("=" * 60)
    
    try:
        # Initialize SQLite Checkpointer
        with SqliteSaver.from_conn_string("trecl_state.db") as memory:
            app = build_graph(checkpointer=memory)
            
            # The config object tracks the thread ID required by the checkpointer
            import uuid
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            
            # For terminal usage, get dynamic input
            company = input("\nEnter startup name to research: ").strip()
            if not company:
                print("No company provided. Exiting.")
                exit(0)
                
            anti_persona = input("Enter your Anti-Persona (e.g., 'No ML research, no model training'): ").strip()
            
            print("\n[*] Initializing Graph Execution...")
            
            # 1. Run until the interrupt
            for event in app.stream({
                "company_name": company,
                "user_domain": "AI Engineer/Applied AI Engineer/AI Agent Developer",
                "user_stack": ["Python", "Langgraph", "RAG", "VectorDB","LLM", "PostgreSQL", "Docker", "FastAPI"],
                "user_anti_persona": anti_persona if anti_persona else "None",
                "company_summary": "",
                "company_jobs": "",
                "github_issues": [],
                "github_prs": [],
                "curated_opportunities": [],
                "selected_targets": [],
                "pain_points_ranked": "",
                "project_ideas": "",
                "cold_email": "",
                "knowledge_store_ready": False
            }, config):
                for node_name, node_state in event.items():
                    print(f"\n\n------------------ Node: {node_name} ------------------")
                    print(f"State: {node_state}\n\n")
            
            # 2. Check if we hit the interrupt
            snapshot = app.get_state(config)
            if snapshot.next and snapshot.next[0] == "pain_synthesizer":
                print("\n" + "=" * 60)
                print("[!] OPPORTUNITIES CURATED. PAUSING FOR HUMAN INPUT.")
                print("=" * 60)
                
                curated_ops = snapshot.values.get("curated_opportunities", [])
                
                if not curated_ops:
                    print("\n[!] No opportunities survived the anti-persona filter. Exiting.")
                    exit(0)
                    
                print("\nAvailable Opportunities:")
                for i, opp in enumerate(curated_ops):
                    print(f"\n[{i+1}] {opp['title']}")
                    print(f"    Tier:      {opp['action_tier']}")
                    print(f"    Type:      {opp['type']}")
                    print(f"    Summary:   {opp.get('description', 'N/A')}")
                    print(f"    URL:       {opp['url']}")
                    print(f"    Relevance: {opp['relevance']}")
                    print(f"    Action:    {opp['suggested_action']}")
                    
                print("\n------------------------------------------------------------")
                selection = input("Enter the numbers of the opportunities to target (e.g., '1, 3'), or 'c' to cancel: ")
                
                if selection.lower() == 'c':
                    print("Execution cancelled.")
                    exit(0)
                    
                selected_indices = [int(i.strip()) - 1 for i in selection.split(",") if i.strip().isdigit()]
                selected_targets = [curated_ops[i] for i in selected_indices if 0 <= i < len(curated_ops)]
                
                if not selected_targets:
                    print("No valid selection made. Proceeding with top item.")
                    selected_targets = [curated_ops[0]]
                    
                print(f"\n[*] Resuming execution with {len(selected_targets)} selected targets...")
                
                # 3. Update the state with selected targets
                app.update_state(config, {"selected_targets": selected_targets})
                
                # 4. Resume execution
                for event in app.stream(None, config):
                    for node_name, node_state in event.items():
                        pass  # Output is already printed inside the nodes
                
                final_state = app.get_state(config).values
                
                print("\n" + "=" * 60)
                print("[>] COMPANY RESEARCH (Extracted Facts)")
                print("=" * 60)
                print(final_state.get("company_summary", "Failed to extract summary."))
                
                print("\n" + "=" * 60)
                print("[>] HIRING SIGNALS (Job Decoder)")
                print("=" * 60)
                print(final_state.get("company_jobs", "Failed to decode jobs."))
                
                print("\n" + "=" * 60)
                print("[>] OPEN SOURCE FOOTPRINT (GitHub Analyst)")
                print("=" * 60)
                issues = final_state.get("github_issues", [])
                prs = final_state.get("github_prs", [])
                if issues:
                    print(f"Issues ({len(issues)}):")
                    for iss in issues:
                        print(f"  - {iss['title']} → {iss['url']}")
                else:
                    print("No open issues found.")
                if prs:
                    print(f"PRs ({len(prs)}):")
                    for pr in prs:
                        print(f"  - {pr['title']} → {pr['url']}")
                
                print("\n" + "=" * 60)
                print("[>] TOP PAIN POINTS (Pain Synthesizer)")
                print("=" * 60)
                print(final_state.get("pain_points_ranked", "Failed to parse pain points."))
                
                print("\n" + "=" * 60)
                print("[>] CUSTOM PROJECT PITCH (Pain Synthesizer)")
                print("=" * 60)
                print(final_state.get("project_ideas", "Failed to parse project ideas."))
                
                print("\n" + "=" * 60)
                print("[@] OUTREACH DRAFTS")
                print("=" * 60)
                print(final_state.get("cold_email", "Failed to generate draft."))
                
    except KeyboardInterrupt:
        print("\nExecution Cancelled by User.")
    except Exception as e:
        print(f"\n[!] Fatal Error: {str(e)}")
