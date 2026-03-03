"""
Trecl Main Entrypoint.
Orchestrates the multi-agent system using LangGraph.
"""

from langgraph.graph import StateGraph, START, END

from core.state import TreclState
from agents.researcher import company_researcher_node
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
    graph.add_node("writer", cold_email_writer_node)
    
    # 3. Define the DAG Execution Order
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "writer")
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
            
        print("\n⏳ Initializing Graph Execution...")
        
        result = app.invoke({
            "company_name": company,
            "company_summary": "",
            "cold_email": ""
        })
        
        print("\n" + "=" * 60)
        print("🔍 COMPANY RESEARCH (Extracted Facts)")
        print("=" * 60)
        print(result.get("company_summary", "Failed to extract summary."))
        
        print("\n" + "=" * 60)
        print("📧 COLD EMAIL DRAFT")
        print("=" * 60)
        print(result.get("cold_email", "Failed to generate draft."))
        
    except KeyboardInterrupt:
        print("\nExecution Cancelled by User.")
    except Exception as e:
        print(f"\n❌ Fatal Error: {str(e)}")
