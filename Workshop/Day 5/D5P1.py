
# workshop/Day 5/D5P1.py

import re
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import TypedDict, Annotated, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send

# ============================================================
# DAY 4 IMPORTS
# ============================================================

project_root = Path(__file__).resolve().parent.parent
day4_dir = project_root / "Day 4"

sys.path.insert(0, str(day4_dir))
sys.path.insert(0, str(project_root))

from D4P2 import make_llm, msg_text, build_agent_graph, calculator

# ============================================================
# TOOLS
# ============================================================

@tool
def get_current_time() -> str:
    """Get today's date and current time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# SHARED STATE
# ============================================================

class MultiState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    route: str
    route_reason: str
    specialist_output: str


# ============================================================
# PROMPTS
# ============================================================

ORCHESTRATOR_PROMPT = """You are the ROUTING layer of a campus multi-agent system.

Decide which specialist should handle the user's question.

Reply with ONE JSON object ONLY:
{"route": "retrieval", "reason": "reason"}
{"route": "general", "reason": "reason"}
{"route": "clarify", "reason": "reason"}

Routing rules:

retrieval:
Questions about THIS college or campus handbook:
library, fees, laboratories, attendance, scholarships,
hostel, exams, regulations, college facilities.

general:
Arithmetic, dates, time, greetings, chitchat,
general knowledge, programming and other general questions.

clarify:
Questions that are ambiguous or underspecified.

Question: {question}

JSON:"""


SYNTH_PROMPT = """You are the synthesis layer of a campus multi-agent system.

The orchestrator routed the question to the '{route}' specialist.

Specialist findings:
{specialist_output}

Original question:
{question}

Rules:
1. Answer using the specialist findings.
2. Preserve handbook citations such as [1] exactly.
3. Never invent or renumber citations.
4. If information is unavailable, say so honestly.
5. If clarification is needed, pass the question through.
6. Keep the answer concise and useful.
"""


# ============================================================
# ROUTE PARSER
# ============================================================

def _parse_route(text: str) -> Optional[dict]:
    """Extract and validate JSON routing output."""

    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)

    if not match:
        return None

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    if data.get("route") not in (
        "retrieval",
        "general",
        "clarify"
    ):
        return None

    data.setdefault("reason", "")
    return data


# ============================================================
# LLM
# ============================================================

def get_gemini():
    """Get the existing Day 4 Gemini model."""
    return make_llm("gemini")


# ============================================================
# ORCHESTRATOR NODE
# ============================================================

def orchestrator_node(state: MultiState):

    question = state["question"]

    llm = get_gemini()

    prompt = ORCHESTRATOR_PROMPT.format(
        question=question
    )

    response = llm.invoke(prompt)

    result = _parse_route(msg_text(response))

    # Safe fallback if routing JSON is invalid
    if result is None:
        route = "general"
        reason = "Fallback because routing output was invalid."
    else:
        route = result["route"]
        reason = result["reason"]

    print("\n[Orchestrator]")
    print("Route:", route)
    print("Reason:", reason)

    return {
        "route": route,
        "route_reason": reason
    }


# ============================================================
# RETRIEVAL SPECIALIST
# ============================================================

def retrieval_node(state: MultiState):

    question = state["question"]

    print("\n[Retrieval Specialist]")
    print("Searching campus knowledge base...")

    try:
        # Reuse Day 4 agent; do not reimplement RAG
        graph = build_agent_graph("gemini")

        result = graph.invoke({
            "messages": [
                HumanMessage(content=question)
            ],
            "tool_call_count": 0,
            "iterations": 0
        })

        answer = msg_text(result["messages"][-1])

    except Exception as e:
        answer = (
            "The retrieval specialist encountered an error: "
            + str(e)
        )

    return {
        "specialist_output": answer
    }


# ============================================================
# GENERAL SPECIALIST
# ============================================================

def general_node(state: MultiState):

    question = state["question"]

    print("\n[General Specialist]")

    # Date/time requests
    date_keywords = [
        "today",
        "current date",
        "current time",
        "date today",
        "what time",
        "what date"
    ]

    if any(word in question.lower() for word in date_keywords):

        answer = get_current_time.invoke({})

        return {
            "specialist_output": answer
        }

    # General questions handled by Gemini
    llm = get_gemini()

    response = llm.invoke([
        SystemMessage(
            content=(
                "You are the general specialist in a campus "
                "multi-agent system. Answer general questions "
                "accurately and clearly."
            )
        ),
        HumanMessage(content=question)
    ])

    answer = msg_text(response)

    return {
        "specialist_output": answer
    }


# ============================================================
# CLARIFICATION SPECIALIST
# ============================================================

def clarify_node(state: MultiState):

    question = state["question"]
    reason = state["route_reason"]

    print("\n[Clarification Specialist]")

    answer = (
        "I need a little more information to answer correctly.\n"
        f"Your question: {question}\n"
        f"Reason: {reason}\n"
        "Could you please clarify what you mean?"
    )

    return {
        "specialist_output": answer
    }


# ============================================================
# SYNTHESIZER
# ============================================================

def synthesizer_node(state: MultiState):

    print("\n[Synthesizer]")

    llm = get_gemini()

    prompt = SYNTH_PROMPT.format(
        route=state["route"],
        specialist_output=state["specialist_output"],
        question=state["question"]
    )

    response = llm.invoke(prompt)

    answer = msg_text(response)

    return {
        "messages": [
            HumanMessage(
                content="Final answer: " + answer
            )
        ]
    }


# ============================================================
# ROUTING
# ============================================================

def route_to_specialist(state: MultiState):

    route = state["route"]

    if route == "retrieval":
        return "retrieval"

    if route == "general":
        return "general"

    return "clarify"


# ============================================================
# BUILD MULTI-AGENT GRAPH
# ============================================================

def build_multiagent_graph():

    workflow = StateGraph(MultiState)

    workflow.add_node(
        "orchestrator",
        orchestrator_node
    )

    workflow.add_node(
        "retrieval",
        retrieval_node
    )

    workflow.add_node(
        "general",
        general_node
    )

    workflow.add_node(
        "clarify",
        clarify_node
    )

    workflow.add_node(
        "synthesizer",
        synthesizer_node
    )

    workflow.add_edge(
        START,
        "orchestrator"
    )

    workflow.add_conditional_edges(
        "orchestrator",
        route_to_specialist,
        {
            "retrieval": "retrieval",
            "general": "general",
            "clarify": "clarify"
        }
    )

    workflow.add_edge(
        "retrieval",
        "synthesizer"
    )

    workflow.add_edge(
        "general",
        "synthesizer"
    )

    workflow.add_edge(
        "clarify",
        "synthesizer"
    )

    workflow.add_edge(
        "synthesizer",
        END
    )

    return workflow.compile()


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    print("=" * 60)
    print("DAY 5 P1 - MULTI-AGENT CAMPUS SYSTEM")
    print("=" * 60)

    app = build_multiagent_graph()

    print("\nMulti-agent graph built successfully.")
    print("Graph nodes:", list(app.get_graph().nodes.keys()))

    print("\nEnter a question.")
    print("Type 'exit' to quit.")

    while True:

        question = input("\nYou: ").strip()

        if question.lower() in ("exit", "quit"):
            print("Exiting Day 5 P1.")
            break

        if not question:
            continue

        initial_state = {
            "messages": [
                HumanMessage(content=question)
            ],
            "question": question,
            "route": "",
            "route_reason": "",
            "specialist_output": ""
        }

        try:

            result = app.invoke(initial_state)

            print("\nAssistant:")
            print(
                msg_text(result["messages"][-1])
            )

        except Exception as e:

            print("\nExecution error:")
            print(e)


# ============================================================
# IMPORT / EXECUTION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
