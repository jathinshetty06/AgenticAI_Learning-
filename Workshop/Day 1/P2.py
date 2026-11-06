
import os
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

client = genai.Client(api_key=GEMINI_API_KEY)

GEMINI_MODEL = "gemini-3.8-flash"
OLLAMA_MODEL = "gemma3:1b"

# Track whether Gemini is currently usable
GEMINI_AVAILABLE = True


# ============================================================
# REUSABLE ASK FUNCTION WITH AUTOMATIC FALLBACK
# ============================================================

def ask(prompt, system="", model=GEMINI_MODEL, temp=0.7):

    global GEMINI_AVAILABLE

    # --------------------------------------------------------
    # Try Gemini
    # --------------------------------------------------------

    if GEMINI_AVAILABLE:

        config = types.GenerateContentConfig(
            temperature=temp,
            system_instruction=system if system else None
        )

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config
            )

            if response.text:
                return response.text.strip() + "\n[Model: Gemini]"

            raise ValueError("Gemini returned an empty response.")

        except Exception as e:

            error_message = str(e)

            if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
                print("\n[Gemini] Free Tier quota exhausted.")
            elif "503" in error_message or "UNAVAILABLE" in error_message:
                print("\n[Gemini] Temporarily unavailable.")
            else:
                print(f"\n[Gemini] Request failed: {e}")

            print("[Fallback] Switching to Ollama.")
            GEMINI_AVAILABLE = False

    # --------------------------------------------------------
    # Ollama fallback
    # --------------------------------------------------------

    try:
        import ollama

        messages = []

        if system:
            messages.append({
                "role": "system",
                "content": system
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={
                "temperature": temp
            }
        )

        answer = response["message"]["content"].strip()

        return answer + "\n[Model: Ollama]"

    except Exception as e:
        print(f"\n[Ollama Error] {e}")

        return "[ERROR] Both Gemini and Ollama are unavailable."


# ============================================================
# TECHNIQUE 1: ZERO-SHOT PROMPTING
# ============================================================

print("\n" + "=" * 60)
print("TECHNIQUE 1: ZERO-SHOT")
print("=" * 60)

zero_shot_prompt = """
Classify the sentiment of this review as Positive, Negative or Neutral.

Review:
"The course materials were well structured but the lab machines were slow."
"""

result = ask(zero_shot_prompt)

print(f"\n[Zero-Shot]\n{result}")


# ============================================================
# TECHNIQUE 2: FEW-SHOT PROMPTING
# ============================================================

print("\n" + "=" * 60)
print("TECHNIQUE 2: FEW-SHOT")
print("=" * 60)

few_shot_prompt = """
Classify a review as Positive, Negative or Neutral.

Examples:

Review: "Absolutely loved the hands-on labs, learned so much!"
Answer: Positive

Review: "Confusing explanations and the code didn't run."
Answer: Negative

Review: "It was okay, nothing special."
Answer: Neutral

Now classify:

Review: "The course materials were well-structured but the lab machines were slow."
Answer:
"""

result = ask(few_shot_prompt, temp=0.0)

print(f"\n[Few-Shot]\n{result}")


# ============================================================
# TECHNIQUE 3A: EXPLICIT CHAIN OF THOUGHT
# ============================================================

print("\n" + "=" * 60)
print("TECHNIQUE 3A: EXPLICIT CoT")
print("=" * 60)

cot_prompt = """
A student submitted 3 assignments worth 20 marks each
and 2 exams worth 50 marks each.

Assignment scores: 18, 15, 17
Exam scores: 42, 38

Calculate the total percentage.

Think step by step:

1. Calculate total assignment marks obtained.
2. Calculate total exam marks obtained.
3. Calculate maximum possible marks.
4. Calculate the percentage using:
   Percentage = (Marks Obtained / Maximum Marks) * 100

Show the calculation clearly.
"""

result = ask(cot_prompt, temp=0.0)

print(f"\n[Explicit CoT]\n{result}")


# ============================================================
# TECHNIQUE 3B: ZERO-SHOT CoT
# ============================================================

print("\n" + "=" * 60)
print("TECHNIQUE 3B: ZERO-SHOT CoT")
print("=" * 60)

zero_shot_cot_prompt = """
A student submitted 3 assignments worth 20 marks each
and 2 exams worth 50 marks each.

Assignment scores: 18, 15, 17
Exam scores: 42, 38

What is the student's total percentage?

Think step by step before answering.
"""

result = ask(zero_shot_cot_prompt, temp=0.0)

print(f"\n[Zero-Shot CoT]\n{result}")


# ============================================================
# TECHNIQUE 4: ROLE PROMPTING
# ============================================================

print("\n" + "=" * 60)
print("TECHNIQUE 4: ROLE PROMPTING")
print("=" * 60)

plain_prompt = """
Should we use a vector database or a relational database
for our RAG system?
"""

plain_result = ask(plain_prompt)

role_result = ask(
    plain_prompt,
    system=(
        "You are a senior database architect with 15 years "
        "of experience in production AI systems. "
        "Give a direct recommendation with engineering "
        "tradeoffs. Be concise."
    )
)

print(f"\n[No Role]\n{plain_result}")

print(f"\n[With Role]\n{role_result}")


# ============================================================
# TECHNIQUE 5: STRUCTURED OUTPUT CONTRACT
# ============================================================

print("\n" + "=" * 60)
print("TECHNIQUE 5: STRUCTURED OUTPUT")
print("=" * 60)

unstructured_prompt = """
Compare Chroma, FAISS, and Pinecone as vector databases.
"""

unstructured_result = ask(unstructured_prompt)

structured_prompt = """
Compare Chroma, FAISS, and Pinecone as vector databases.

Return your answer as a markdown table with exactly
these columns:

| Database | Type | Best For | Limitation |

Do not add text before or after the table.
"""

structured_result = ask(structured_prompt, temp=0.0)

print("\n[No Format]")
print(unstructured_result[:300] + "...")

print("\n[With Format Contract]")
print(structured_result)


# ============================================================
# TECHNIQUE 6: SELF-CONSISTENCY
# ============================================================

print("\n" + "=" * 60)
print("TECHNIQUE 6: SELF-CONSISTENCY")
print("=" * 60)

reasoning_prompt = """
A RAG pipeline retrieves 5 chunks for a question.

Chunk relevance scores:
[0.91, 0.88, 0.72, 0.65, 0.41]

Should we use the top-3 or top-5 chunks?

Give exactly one answer:
top-3 or top-5
"""

votes = []

for i in range(5):

    answer = ask(reasoning_prompt, temp=0.7)

    # Remove model labels before processing
    clean_answer = answer.lower()

    # Extract only valid choices
    match = re.search(r"\btop[- ]?[35]\b", clean_answer)

    if match:

        choice = match.group().replace(" ", "-")

        if choice == "top-3" or choice == "top3":
            votes.append("top-3")

        elif choice == "top-5" or choice == "top5":
            votes.append("top-5")

        else:
            votes.append("Invalid")

    else:
        votes.append("Invalid")

valid_votes = [v for v in votes if v in ["top-3", "top-5"]]

if valid_votes:

    majority = max(
        set(valid_votes),
        key=valid_votes.count
    )

else:
    majority = "No valid majority"

print(f"\n[Self-Consistency]")
print(f"Votes: {votes}")
print(f"Majority: {majority}")


# ============================================================
# TECHNIQUE 7: SYSTEM INSTRUCTION VS USER MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("TECHNIQUE 7: SYSTEM INSTRUCTION")
print("=" * 60)

SYSTEM = """
You are a technical documentation assistant
for an engineering course.

Rules:
1. Always answer in under 3 sentences.
2. Use precise technical vocabulary.
3. If the question is outside your domain,
   say 'Out of scope.'
"""

questions = [
    "What is HNSW and why does Chroma use it?",
    "What is the best recipe for biryani?",
    "What does MMR do in a retrieval pipeline?"
]

for q in questions:

    answer = ask(q, system=SYSTEM)

    print(f"\nQ: {q}")
    print(f"A: {answer}")


# ============================================================
# TECHNIQUE 8: CONTEXT WINDOW BUDGETING
# ============================================================

print("\n" + "=" * 60)
print("TECHNIQUE 8: CONTEXT WINDOW BUDGETING")
print("=" * 60)

retrieved_chunks = [

    "Chroma uses HNSW (Hierarchical Navigable Small World) "
    "graphs for approximate nearest-neighbour search.",

    "FAISS is a library developed by Meta AI for efficient "
    "similarity search and clustering of dense vectors. " * 8,

    "Pinecone is a managed vector database that handles "
    "infrastructure, scaling, and replication automatically.",

    "MMR (Maximal Marginal Relevance) balances relevance "
    "and diversity when selecting retrieval results.",

    "Cosine similarity measures the angle between two vectors. "
    "Values range from -1 to 1. " * 5
]

MAX_CONTEXT_CHARS = 800

QUESTION = "What is HNSW and how does it relate to Chroma?"

packed_chunks = []
total = 0

for chunk in retrieved_chunks:

    remaining = MAX_CONTEXT_CHARS - total

    if remaining <= 0:
        break

    if len(chunk) <= remaining:

        packed_chunks.append(chunk)
        total += len(chunk)

    else:

        if remaining > 100:

            truncated_chunk = chunk[:remaining - 1] + "…"

            packed_chunks.append(truncated_chunk)

            total += len(truncated_chunk)

        break

context_block = "\n\n".join(
    f"[{i + 1}] {chunk}"
    for i, chunk in enumerate(packed_chunks)
)

budgeted_prompt = f"""
You are a helpful AI assistant.

Answer the question using ONLY the context below.
If the answer is not in the context, say:
"I don't know."

Context:
{context_block}

Question: {QUESTION}

Answer:
"""

answer = ask(budgeted_prompt, temp=0.0)

print("\n[Budgeted RAG Prompt]")
print(f"Context chars used: {total}")
print(f"Answer: {answer}")


# ============================================================
# PROGRAM COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("P2 PROGRAM COMPLETED SUCCESSFULLY")
print("=" * 60)
