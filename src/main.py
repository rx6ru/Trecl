"""
Trecl Main Entrypoint.
Orchestrates the multi-agent system using LangGraph.
"""

from langgraph.graph import StateGraph, START, END

from core.state import TreclState
from agents.researcher import company_researcher_node
from agents.job_decoder import job_decoder_node
from agents.pain_synthesizer import pain_synthesizer_node
from agents.writer import cold_email_writer_node

def build_graph():
    """
    Constructs the LangGraph execution flow.
    
    Nodes:
        - researcher: Scrapes the web and extracts facts.
        - writer: Drafts a cold email using those facts.
        
    Flow:
        START -> researcher -> writer -> END
        
    Returns:
        CompiledGraph: The executable state machine.
    """
    # 1. Initialize Graph with central State
    graph = StateGraph(TreclState)
    
    # 2. Add Nodes (Agents)
    graph.add_node("researcher", company_researcher_node)
    graph.add_node("job_decoder", job_decoder_node)
    graph.add_node("pain_synthesizer", pain_synthesizer_node)
    graph.add_node("writer", cold_email_writer_node)
    
    # 3. Define the DAG Execution Order (Parallel Fan-Out / Fan-In)
    graph.add_edge(START, "researcher")
    graph.add_edge(START, "job_decoder")
    
    graph.add_edge("researcher", "pain_synthesizer")
    graph.add_edge("job_decoder", "pain_synthesizer")
    
    graph.add_edge("pain_synthesizer", "writer")
    graph.add_edge("writer", END)
    
    # 4. Compile into an executable app
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    
    print("=" * 60)
    print("TRECL INTELLIGENCE AGENT (v0.2)")
    print("=" * 60)
    
    try:
        # For terminal usage, get dynamic input
        company = input("\nEnter startup name to research: ").strip()
        
        if not company:
            print("No company provided. Exiting.")
            exit(0)
            
        print("\n[*] Initializing Graph Execution...")
        
        result = app.invoke({
            "company_name": company,
            "user_domain": "Backend Engineer",
            "user_stack": ["Python", "Go", "PostgreSQL", "Docker"],
            "company_summary": "",
            "company_jobs": "",
            "pain_points_ranked": "",
            "project_ideas": "",
            "cold_email": ""
        })
        
        print("\n" + "=" * 60)
        print("[>] COMPANY RESEARCH (Extracted Facts)")
        print("=" * 60)
        print(result.get("company_summary", "Failed to extract summary."))
        
        print("\n" + "=" * 60)
        print("[>] HIRING SIGNALS (Job Decoder)")
        print("=" * 60)
        print(result.get("company_jobs", "Failed to decode jobs."))
        
        print("\n" + "=" * 60)
        print("[>] TOP PAIN POINTS (Pain Synthesizer)")
        print("=" * 60)
        print(result.get("pain_points_ranked", "Failed to parse pain points."))
        
        print("\n" + "=" * 60)
        print("[>] CUSTOM PROJECT PITCH (Pain Synthesizer)")
        print("=" * 60)
        print(result.get("project_ideas", "Failed to parse project ideas."))
        
        print("\n" + "=" * 60)
        print("[@] COLD EMAIL DRAFT")
        print("=" * 60)
        print(result.get("cold_email", "Failed to generate draft."))
        
    except KeyboardInterrupt:
        print("\nExecution Cancelled by User.")
    except Exception as e:
        print(f"\n[!] Fatal Error: {str(e)}")
