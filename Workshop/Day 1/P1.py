
import os
import time
import json

from dotenv import load_dotenv
from google import genai
from google.genai import types
import ollama

# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY not found")

client = genai.Client(api_key=GEMINI_API_KEY)

GEMINI_MODEL = "gemini-3.6-flash"
OLLAMA_MODEL = "gemma3:1b"


# ============================================================
# PING TEST
# ============================================================

print("\n--- API PING ---", flush=True)

try:
    ping = client.models.generate_content(
        model=GEMINI_MODEL,
        contents="Reply with exactly three words: API key works."
    )

    print(f"[PING] {ping.text.strip()}", flush=True)

except Exception as e:
    print(f"[PING ERROR] {e}", flush=True)


# ============================================================
# NORMAL GENERATION
# ============================================================

BENCHMARK_PROMPT = (
    "Explain what a Transformer neural network is in exactly 3 sentences"
)

print("\n--- NORMAL GENERATION ---", flush=True)

# ---------------- GEMINI ----------------

print("\n[1] Starting Gemini generation...", flush=True)

t0 = time.perf_counter()

try:
    gemini_resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=BENCHMARK_PROMPT
    )

    gemini_latency = time.perf_counter() - t0
    gemini_text = gemini_resp.text.strip()
    gemini_tokens = gemini_resp.usage_metadata.total_token_count

    print("[1] Gemini completed.", flush=True)

except Exception as e:
    gemini_latency = None
    gemini_text = "Unavailable"
    gemini_tokens = "N/A"

    print(f"[1] Gemini failed: {e}", flush=True)


# ---------------- OLLAMA ----------------

print("\n[2] Starting Ollama generation...", flush=True)

t0 = time.perf_counter()

try:
    ollama_resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": BENCHMARK_PROMPT}]
    )

    ollama_latency = time.perf_counter() - t0
    ollama_text = ollama_resp["message"]["content"].strip()
    ollama_tokens = ollama_resp.get("eval_count", "N/A")

    print("[2] Ollama completed.", flush=True)

except Exception as e:
    ollama_latency = None
    ollama_text = "Unavailable"
    ollama_tokens = "N/A"

    print(f"[2] Ollama failed: {e}", flush=True)


# ---------------- DISPLAY RESULTS ----------------

print(f"\n{'MODEL':<25}{'Latency(s)':>12}{'Tokens':>10}")

print(
    f"{GEMINI_MODEL:<25}"
    f"{f'{gemini_latency:.2f}' if gemini_latency is not None else 'N/A':>12}"
    f"{str(gemini_tokens):>10}"
)

print(
    f"{OLLAMA_MODEL:<25}"
    f"{f'{ollama_latency:.2f}' if ollama_latency is not None else 'N/A':>12}"
    f"{str(ollama_tokens):>10}"
)

print(f"\nGemini result:\n{gemini_text}")
print(f"\nOllama result:\n{ollama_text}")


# ============================================================
# STREAMING
# ============================================================

STREAM_PROMPT = (
    "List 5 real-world use cases of Retrieval-Augmented Generation, "
    "one per line."
)


# ---------------- GEMINI STREAMING ----------------

print("\n--- GEMINI STREAMING ---", flush=True)

try:
    for chunk in client.models.generate_content_stream(
        model=GEMINI_MODEL,
        contents=STREAM_PROMPT
    ):
        if chunk.text:
            print(chunk.text, end="", flush=True)

    print("\n[Gemini Streaming Completed]")

except Exception as e:
    print(f"\n[Gemini Streaming Error] {e}")
    print("[Skipping Gemini streaming...]")

print()


# ---------------- OLLAMA STREAMING ----------------

print("\n--- OLLAMA STREAMING ---", flush=True)

try:
    for chunk in ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": STREAM_PROMPT}],
        stream=True
    ):
        token = chunk["message"]["content"]
        print(token, end="", flush=True)

    print("\n[Ollama Streaming Completed]")

except Exception as e:
    print(f"\n[Ollama Streaming Error] {e}")

print()


# ============================================================
# JSON MODE
# ============================================================

JSON_PROMPT = """
Summarize the following paragraph.

Return a JSON object with exactly these keys:
- "summary": one sentence summary (string)
- "topics": list of up to 3 key topics (array of strings)
- "sentiment": one of "positive", "neutral", or "negative" (string)

Paragraph:
"Retrieval-Augmented Generation (RAG) is a powerful technique that combines
the strengths of dense retrieval with large language models. By grounding
LLM responses in retrieved documents, RAG dramatically reduces
hallucinations and keeps answers factual. It has become the dominant
architecture for enterprise knowledge-base chatbots and Q&A systems."
"""

print("\n--- GEMINI JSON MODE ---", flush=True)

try:
    json_resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=JSON_PROMPT,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1
        )
    )

    structured = json.loads(json_resp.text)

    print("\n--- JSON MODE OUTPUT ---")
    print(f"Summary  : {structured['summary']}")
    print(f"Topics   : {structured['topics']}")
    print(f"Sentiment: {structured['sentiment']}")

except Exception as e:
    print(f"[Gemini JSON Error] {e}")


# ---------------- OLLAMA JSON MODE ----------------

print("\n--- OLLAMA JSON MODE ---", flush=True)

try:
    ollama_json_resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": JSON_PROMPT + "\nReturn ONLY valid JSON."
            }
        ],
        format="json"
    )

    ollama_structured = json.loads(
        ollama_json_resp["message"]["content"]
    )

    print(f"Summary  : {ollama_structured['summary']}")
    print(f"Topics   : {ollama_structured['topics']}")
    print(f"Sentiment: {ollama_structured['sentiment']}")

except Exception as e:
    print(f"[Ollama JSON Error] {e}")


# ============================================================
# 5-RUN LATENCY BENCHMARK
# ============================================================

BENCH_PROMPT = (
    "What is the difference between fine-tuning and RAG? "
    "Answer in 2 sentences."
)

N_RUNS = 5

print(f"\n--- {N_RUNS}-RUN LATENCY BENCHMARK ---", flush=True)

gemini_times = []
ollama_times = []

last_gemini_response = None

for i in range(1, N_RUNS + 1):

    # ---------------- GEMINI ----------------

    print(f"\nRun {i}: Gemini...", flush=True)

    t0 = time.perf_counter()

    try:
        r = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=BENCH_PROMPT
        )

        elapsed = time.perf_counter() - t0

        gemini_times.append(elapsed)
        last_gemini_response = r

        gemini_run = f"{elapsed:.2f}s"

    except Exception as e:
        gemini_times.append(None)
        gemini_run = "N/A"

        print(f"Gemini failed: {e}", flush=True)

    # ---------------- OLLAMA ----------------

    print(f"Run {i}: Ollama...", flush=True)

    t0 = time.perf_counter()

    try:
        ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": BENCH_PROMPT}]
        )

        elapsed = time.perf_counter() - t0

        ollama_times.append(elapsed)
        ollama_run = f"{elapsed:.2f}s"

    except Exception as e:
        ollama_times.append(None)
        ollama_run = "N/A"

        print(f"Ollama failed: {e}", flush=True)

    print(
        f"Run {i}: Gemini={gemini_run} | Ollama={ollama_run}",
        flush=True
    )


# ============================================================
# AVERAGE LATENCY
# ============================================================

valid_gemini_times = [
    t for t in gemini_times if t is not None
]

valid_ollama_times = [
    t for t in ollama_times if t is not None
]

print("\n--- FINAL BENCHMARK RESULTS ---")

if valid_gemini_times:
    gemini_avg = sum(valid_gemini_times) / len(valid_gemini_times)

    print(
        f"Average Gemini: {gemini_avg:.2f}s "
        f"({len(valid_gemini_times)}/{N_RUNS} successful runs)"
    )
else:
    print("Average Gemini: N/A")

if valid_ollama_times:
    ollama_avg = sum(valid_ollama_times) / len(valid_ollama_times)

    print(
        f"Average Ollama: {ollama_avg:.2f}s "
        f"({len(valid_ollama_times)}/{N_RUNS} successful runs)"
    )
else:
    print("Average Ollama: N/A")


# ============================================================
# TOKEN USAGE
# ============================================================

if last_gemini_response:
    print(
        "\nLast Gemini token count:",
        last_gemini_response.usage_metadata.total_token_count,
        "tokens"
    )
else:
    print("\nLast Gemini token count: N/A")


# ============================================================
# COMPLETED
# ============================================================

print("\n--- P1 PROGRAM COMPLETED ---")
