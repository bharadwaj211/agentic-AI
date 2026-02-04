import os
import langchain
import operator
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# API Key for the LLM
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

# Hardcoding the weather data parameters for various cities
Weather_database = {
    "Mumbai": {"temperature": 15, "condition": "cloudy", "humidity": 70, "Wind_Kmh": 18},
    "Bengaluru": {"temperature": 22, "condition": "partly cloudy", "humidity": 58, "Wind_Kmh": 10},
    "Hyderabad": {"temperature": 32, "condition": "breezy", "humidity": 25, "Wind_Kmh": 23},
    "Delhi": {"temperature": 25, "condition": "Rainy", "humidity": 70, "Wind_Kmh": 45},
    "Pune": {"temperature": 35, "condition": "Hot", "humidity": 70, "Wind_Kmh": 35},
    "Jaipur": {"temperature": 14, "condition": "Dry", "humidity": 35, "Wind_Kmh": 9},
    "Ahmedabad": {"temperature": 23, "condition": "clear", "humidity": 45, "Wind_Kmh": 13},
    "Lucknow": {"temperature": 13, "condition": "foggy", "humidity": 72, "Wind_Kmh": 7},
    "Nagpur": {"temperature": 21, "condition": "sunny", "humidity": 72, "Wind_Kmh": 18},
    "Goa": {"temperature": 26, "condition": "clear", "humidity": 48, "Wind_Kmh": 10},
    "Kochi": {"temperature": 27, "condition": "Rain Showers", "humidity": 82, "Wind_Kmh": 20},
}

# Calling the weather data for the cities
@tool
def get_weather(city: str) -> str:
    """Return weather information for a given city."""
    city = city.title()  

    if city not in Weather_database:
        return f"Weather data for {city} is not available."

    data = Weather_database[city]
    return (
        f"Weather in {city}:\n"
        f"Temperature: {data['temperature']}°C\n"
        f"Condition: {data['condition']}\n"
        f"Humidity: {data['humidity']}%\n"
        f"Wind: {data['Wind_Kmh']} km/h"
    )

tools = [get_weather]

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# Details of the LLM Node
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=200  
).bind_tools(tools)

def llm_node(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# LangGraph
graph = StateGraph(AgentState)
graph.add_node("llm", llm_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("llm")
graph.add_conditional_edges("llm", should_continue)
graph.add_edge("tools", "llm")
app = graph.compile()

# Execution of the script to know the weather conditions for the city entered
if __name__ == "__main__":
    city = input("Enter city name: ")

    user_message = HumanMessage(
        content=f"What is the weather in {city}?"
    )

    result = app.invoke({"messages": [user_message]})

    print("\n=== WEATHER REPORT ===")
    print(result["messages"][-1].content)
