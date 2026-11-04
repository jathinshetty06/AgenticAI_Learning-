import json
import ollama
from tools import TOOL_REGISTRY

MODEL = "gemma3:4b"
MAX_ITERATIONS = 5

# Tool descriptions for Gemma
TOOL_DESCRIPTIONS = {
    "get_weather": {
        "description": "Get simulated weather for a city.",
        "arguments": {"city": "city name"}
    },
    "calculator": {
        "description": "Evaluate a mathematical expression.",
        "arguments": {"expression": "math expression"}
    },
    "get_current_time": {
        "description": "Get current date and time.",
        "arguments": {"timezone": "timezone name, optional"}
    }
}

def dispatch_tool(name, args):
    """Execute a registered tool safely and return its result."""
    if name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {name}"}
    try:
        return TOOL_REGISTRY[name](**args)
    except Exception as exc:
        return {"error": f"Tool {name} failed: {exc}"}
    
def run_agent(user_query, max_iterations=MAX_ITERATIONS):
    print("\n" + "-" * 60)
    print(f"[User] {user_query}")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful tool-using AI agent.\n"
                "Available tools:\n"
                + json.dumps(TOOL_DESCRIPTIONS, indent=2)
                + "\nWhen a tool is needed, respond ONLY with valid JSON in this format:\n"
                '{"tool":"tool_name","args":{"argument":"value"}}\n'
                "If no tool is needed or after receiving tool results, answer the user normally. "
                "Never invent tool results."
            )
        },
        {"role": "user", "content": user_query}
    ]

    for iteration in range(1, max_iterations + 1):
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            options={"temperature": 0}
        )
        text = response["message"]["content"].strip()
        messages.append({"role": "assistant", "content": text})

        # Try to parse a tool request from Gemma's response
        tool_call = None
        try:
            cleaned = text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and "tool" in parsed:
                tool_call = parsed
        except json.JSONDecodeError:
            pass

        if not tool_call:
            print(f"[Final Answer] {text}")
            return text

        name = tool_call.get("tool")
        args = tool_call.get("args", {})
        if not isinstance(args, dict):
            args = {}

        print(f"[Iter {iteration}] Tool call: {name}({args})")
        result = dispatch_tool(name, args)
        observation = json.dumps(result, ensure_ascii=False, default=str)
        print(f"[Iter {iteration}] Observation: {observation}")

        messages.append({
            "role": "user",
            "content": f"Tool result for {name}: {observation}\nNow answer the original question, or request another tool using the JSON format."
        })

    return "[Max iterations reached]"

TEST_QUERIES = [
    "What is the weather like in Bangalore right now?",
    "What is 2 to the power of 10?",
    "What is today's date and time?",
    "If I invest 50000 at 8% annual simple interest for 3 years, how much interest do I earn?"
]

if __name__ == "__main__":
    print("=" * 60)
    print("  OLLAMA TOOL-CALLING AGENT (GEMMA 3:4B)")
    print("=" * 60)

    for query in TEST_QUERIES:
        try:
            run_agent(query)
        except Exception as exc:
            print(f"[Agent Error] {exc}")
