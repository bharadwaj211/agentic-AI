import os
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# API Key for the LLM
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

# Defining the agent state on the research topic
class ResearchState(TypedDict):
    topic: str
    research_output: str

# LLM Configuration details for the node
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=200
)

# Prompt Template for answering the queries posted by the user
research_prompt = PromptTemplate.from_template("""
You are a Research Assistant Agent.

Conduct detailed research on the following topic:

Topic: {topic}

Provide the output in a structured format:

1. Overview
2. Key Points
3. Applications / Examples
4. Conclusion
""")

# Research Node 
def research_node(state: ResearchState) -> ResearchState:
    prompt = research_prompt.format(topic=state["topic"])
    response = llm.invoke(prompt)

    return {
        "topic": state["topic"],
        "research_output": response.content
    }

# Build LangGraph
graph = StateGraph(ResearchState)

graph.add_node("research_agent", research_node)
graph.set_entry_point("research_agent")
graph.add_edge("research_agent", END)

research_graph = graph.compile()

# Execution of the logic to see the output
if __name__ == "__main__":
    topic = input("Enter research topic: ")

    result = research_graph.invoke({
        "topic": topic
    })

    print("\n===== Structured Research Summary =====\n")
    print(result["research_output"])
