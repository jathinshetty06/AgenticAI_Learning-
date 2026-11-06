
# ------------------------------------------------------------
# DAY 4 P2 - LANGGRAPH AGENT
# ------------------------------------------------------------

import sys
from pathlib import Path
from typing import TypedDict, Annotated, List, Dict, Any

from dotenv import load_dotenv

# ------------------------------------------------------------
# PROJECT PATH SETUP
# ------------------------------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent
WORKSHOP_DIR = CURRENT_DIR.parent
PROJECT_ROOT = WORKSHOP_DIR.parent

DAY3_DIR = WORKSHOP_DIR / "Day 3"
DAY2_DIR = WORKSHOP_DIR / "Day 2"

# Support repositories using Day3 / Day2 without spaces too
if not DAY3_DIR.is_dir():
    DAY3_DIR = WORKSHOP_DIR / "Day3"

if not DAY2_DIR.is_dir():
    DAY2_DIR = WORKSHOP_DIR / "Day2"

# Add all required paths
for folder in [
    PROJECT_ROOT,
    WORKSHOP_DIR,
    DAY2_DIR,
    DAY3_DIR,
]:
    if folder.is_dir():
        folder_str = str(folder)
        if folder_str not in sys.path:
            sys.path.insert(0, folder_str)

# ------------------------------------------------------------
# ENVIRONMENT
# ------------------------------------------------------------

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(CURRENT_DIR / ".env")

# ------------------------------------------------------------
# LANGGRAPH IMPORTS
# ------------------------------------------------------------

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# ------------------------------------------------------------
# LANGCHAIN IMPORTS
# ------------------------------------------------------------

from langchain_core.tools import tool
from langchain_core.messages import (
    AIMessage,
    ToolMessage,
    HumanMessage,
    SystemMessage,
)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

# ------------------------------------------------------------
# DAY 3 RAG ENGINE
# ------------------------------------------------------------

from D3P3 import answer as rag_answer

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

GEMINI_MODEL = "gemini-3.6-flash"

# Correct model installed in Ollama
OLLAMA_MODEL = "gemma3:4b"

MAX_ITERATIONS = 5

# ------------------------------------------------------------
# MESSAGE TEXT EXTRACTION
# ------------------------------------------------------------

def msg_text(message: Any) -> str:
    """
    Extract readable text from a LangChain message.
    """

    content = getattr(message, "content", message)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []

        for block in content:
            if isinstance(block, dict):
                if block.get("type") in (None, "text"):
                    parts.append(block.get("text", ""))

            else:
                text = getattr(block, "text", None)

                if text:
                    parts.append(text)

        return "".join(parts)

    return str(content)

# ------------------------------------------------------------
# LLM FACTORY
# ------------------------------------------------------------

def make_llm(brain: str = "gemini"):
    """
    Create the selected LLM.

    Supported:
        gemini
        ollama
    """

    if brain == "gemini":
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL
        )

    if brain == "ollama":
        return ChatOllama(
            model=OLLAMA_MODEL
        )

    raise ValueError(
        f"Unknown brain '{brain}'. "
        "Choose: gemini | ollama"
    )

# ------------------------------------------------------------
# KNOWLEDGE BASE TOOL
# ------------------------------------------------------------

@tool
def search_knowledge_base(question: str) -> Dict[str, Any]:
    """
    Search the campus handbook knowledge base.

    Use for college facts including:
    library, fees, labs, attendance, scholarships,
    hostel, and examinations.
    """

    try:
        result = rag_answer(
            question,
            backend="chroma",
            llm="gemini",
            topk=5,
        )

        return {
            "answer": result["text"],
            "citations": result["citations"],
            "abstained": result["abstained"],
        }

    except Exception as e:
        return {
            "error": f"Search failed: {e}"
        }

# ------------------------------------------------------------
# CALCULATOR TOOL
# ------------------------------------------------------------

@tool
def calculator(expression: str) -> Dict[str, Any]:
    """
    Calculate mathematical expressions.

    Supports:
        + - * / **
        parentheses
        abs()
        round()

    Example:
        25 * 4
    """

    try:
        value = eval(
            expression,
            {"__builtins__": {}},
            {
                "abs": abs,
                "round": round,
            },
        )

        return {
            "result": round(float(value), 6)
        }

    except Exception as e:
        return {
            "error": f"Could not evaluate '{expression}': {e}"
        }

# ------------------------------------------------------------
# AGENT TOOLS
# ------------------------------------------------------------

AGENT_TOOLS = [
    search_knowledge_base,
    calculator,
]

# ------------------------------------------------------------
# SYSTEM PROMPT
# ------------------------------------------------------------

SYSTEM_PROMPT = """You are a campus knowledge agent.

Rules:
- For any college-related fact, use search_knowledge_base.
- For arithmetic, use calculator.
- Call ONE tool at a time.
- If search returns abstained=true, do not search again.
- If information is unavailable, clearly say so.
- Cite handbook facts using returned citation markers.
- Never invent college rules, fees, or dates.
"""

# ------------------------------------------------------------
# AGENT STATE
# ------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_call_count: int
    iterations: int

# ------------------------------------------------------------
# BUILD AGENT GRAPH
# ------------------------------------------------------------

def build_agent_graph(
    brain: str = "gemini",
    checkpointer=None,
):
    """
    Build the LangGraph agent.

    Graph:
        START -> agent -> tools -> agent -> END
    """

    # Bind tools to the selected model
    llm_with_tools = make_llm(brain).bind_tools(
        AGENT_TOOLS
    )

    # --------------------------------------------------------
    # AGENT NODE
    # --------------------------------------------------------

    def agent_node(state: AgentState) -> dict:

        response = llm_with_tools.invoke(
            [
                SystemMessage(
                    content=SYSTEM_PROMPT
                )
            ]
            + state["messages"]
        )

        return {
            "messages": [response],

            "tool_call_count":
                state.get("tool_call_count", 0)
                + len(response.tool_calls),

            "iterations":
                state.get("iterations", 0) + 1,
        }

    # --------------------------------------------------------
    # CONDITIONAL ROUTING
    # --------------------------------------------------------

    def should_continue(state: AgentState) -> str:

        # Stop if iteration limit reached
        if state.get("iterations", 0) >= MAX_ITERATIONS:
            return "__end__"

        last_message = state["messages"][-1]

        # Continue if model requested a tool
        if getattr(
            last_message,
            "tool_calls",
            None,
        ):
            return "tools"

        # Otherwise finish
        return "__end__"

    # --------------------------------------------------------
    # GRAPH BUILDER
    # --------------------------------------------------------

    builder = StateGraph(AgentState)

    builder.add_node(
        "agent",
        agent_node,
    )

    builder.add_node(
        "tools",
        ToolNode(AGENT_TOOLS),
    )

    builder.add_edge(
        START,
        "agent",
    )

    builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "__end__": END,
        },
    )

    builder.add_edge(
        "tools",
        "agent",
    )

    # --------------------------------------------------------
    # COMPILE
    # --------------------------------------------------------

    return builder.compile(
        checkpointer=checkpointer
    )

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 55)
    print("DAY 4 P2 - LANGGRAPH AGENT")
    print("=" * 55)

    print("Project root:", PROJECT_ROOT)
    print("Workshop:", WORKSHOP_DIR)
    print("Day 2:", DAY2_DIR)
    print("Day 3:", DAY3_DIR)

    print("\nTesting Day 3 RAG import...")
    print("RAG engine imported successfully.")

    print("\nTesting Ollama...")
    print(
        msg_text(
            make_llm("ollama").invoke(
                "Reply with: Ollama connection successful"
            )
        )
    )

    print("\nBuilding LangGraph agent...")
    graph = build_agent_graph("ollama")

    print(
        "Graph nodes:",
        list(graph.get_graph().nodes),
    )

    print("\nDay 4 P2 setup completed.")
