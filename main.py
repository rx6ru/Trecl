from dotenv import load_dotenv

load_dotenv()

from typing import TypedDict

from langchain_cerebras import ChatCerebras
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

llm = ChatCerebras(model="gpt-oss-120b")


class Trecl(TypedDict):
    company_name: str
    company_summary: str
    cold_email: str


def company_researcher(state: Trecl) -> dict:
    print(f"🕵️‍♂️ Researching {state['company_name']}...")
    message = [
        SystemMessage(
            content="You are a Deep research Agent. You are currently under test - So generate Dumy Data from your knowledge or make something up if needed. TASK - Find about the given/mentioned company/startup, and output a very short, plain text, single paragraph summary about them"
        ),
        HumanMessage(content="Find something about Zepto"),
    ]
    response = llm.invoke(message)
    return {"company_summary": response.content}


def cold_email_writer(state: Trecl) -> dict:
    print("✍️ Writing the email...")
    message = [
        SystemMessage(
            content="You are a Smart Writer Agent. You are currently under test - So generate Dumy Data from your knowledge or make something up is needed. TASK - Write a cold email about the given company, by reading its short summary -  "
        ),
        AIMessage(content=state["company_summary"]),
    ]
    response = llm.invoke(message)
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
