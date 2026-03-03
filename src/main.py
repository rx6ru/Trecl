from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from LLMs.cerebras import CEREBRAS
from tools.tavily import TRAVILY


class Trecl(TypedDict):
    company_name: str
    company_summary: str
    cold_email: str


def company_researcher(state: Trecl) -> dict:
    company = state["company_name"]

    print(f"🕵️‍♂️ Researching {company}...")

    search_results = TRAVILY.search(
        query=f"{company} startup what they do product tech stacl", max_results=3
    )

    message = [
        SystemMessage(
            content="You are a Deep research Agent. Use the Search Results and output a short crips plain text summary"
        ),
        HumanMessage(
            content=f"Company Name = {state['company_name']}\nSearch Results  = {search_results}"
        ),
    ]
    response = CEREBRAS.invoke(message)
    return {"company_summary": response.content}


def cold_email_writer(state: Trecl) -> dict:
    print("✍️ Writing the email...")
    message = [
        SystemMessage(
            content="You are a Smart Writer Agent. You are currently under test - So generate Dumy Data from your knowledge or make something up is needed. TASK - Write a cold email about the given company, by reading its short summary -  "
        ),
        AIMessage(content=f"Company Name = {state['company_summary']}"),
    ]
    response = CEREBRAS.invoke(message)
    return {"cold_email": response.content}


def build_graph():
    graph = StateGraph(Trecl)

    graph.add_node("researcher", company_researcher)
    graph.add_node("writer", cold_email_writer)

    graph.add_edge(START, "researcher")

    graph.add_edge("researcher", "writer")

    graph.add_edge("writer", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    result = app.invoke(
        {"company_name": "Zepto", "company_summary": "", "cold_email": ""}
    )

    print("\n--- FINAL STATE ---")
    print(result)
