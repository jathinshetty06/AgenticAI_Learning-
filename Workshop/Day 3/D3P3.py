import os
import re
import sys
import time
import random
import argparse

from pathlib import Path
from typing import List, Dict, Any, Optional, Generator

import ollama
from dotenv import load_dotenv
from google import genai


# ------------------------------------------------------------
# PROJECT PATH SETUP
# ------------------------------------------------------------


# ------------------------------------------------------------
# PROJECT PATH SETUP
# ------------------------------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent
WORKSHOP_DIR = CURRENT_DIR.parent
PROJECT_ROOT = WORKSHOP_DIR.parent

DAY2_DIR = WORKSHOP_DIR / "Day 2"
DAY3_DIR = WORKSHOP_DIR / "Day 3"

for folder in (
    DAY2_DIR,
    DAY3_DIR,
    WORKSHOP_DIR,
    PROJECT_ROOT,
):
    if folder.is_dir() and str(folder) not in sys.path:
        sys.path.insert(0, str(folder))


# ------------------------------------------------------------
# IMPORT DAY 2 AND DAY 3 MODULES
# ------------------------------------------------------------

from D2P2 import get_vector_store

from D3P1 import (
    rewrite_query,
    mmr_rerank,
    gemini_text,
    embed_query_ollama,
    embed_ollama,
    GEMINI_CHAT_MODEL,
    OLLAMA_CHAT_MODEL,
)

from D3P2 import build_prompt



# ------------------------------------------------------------
# CORRECTED IMPORTS
# ------------------------------------------------------------

# Add actual Day 2 folder to Python import path
DAY2_DIR = WORKSHOP_DIR / "Day 2"

if not DAY2_DIR.is_dir():
    raise FileNotFoundError(f"Day 2 folder not found: {DAY2_DIR}")

if str(DAY2_DIR) not in sys.path:
    sys.path.insert(0, str(DAY2_DIR))

# Import Day 2 module
from D2P2 import get_vector_store

from D3P1 import (
    rewrite_query,
    mmr_rerank,
    gemini_text,
    embed_query_ollama,
    embed_ollama,
    GEMINI_CHAT_MODEL,
    OLLAMA_CHAT_MODEL,
)
from D3P2 import build_prompt


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

load_dotenv()

DEFAULT_COLLECTION = "capstone_corpus"
FETCH_MULTIPLIER = 3

CITE_RE = re.compile(r"\[(\d+)\]")

ABSTAIN_MARKERS = (
    "not in the knowledge base",
    "i don't know",
    "i do not know",
)

client = None


def get_gemini_client():
    global client

    if client is None:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing. Check your .env file."
            )

        client = genai.Client(api_key=api_key)

    return client


# ------------------------------------------------------------
# RAG PIPELINE
# ------------------------------------------------------------

def _prepare(
    question: str,
    backend: str,
    topk: int,
    mmr: bool,
    rewrite: bool,
    collection: str,
) -> Dict[str, Any]:

    latency = {}

    # Step 1: Query rewriting
    t0 = time.perf_counter()

    search_q = rewrite_query(question) if rewrite else question

    latency["rewrite_s"] = (
        round(time.perf_counter() - t0, 3)
        if rewrite else 0.0
    )

    # Step 2: Query embedding
    t0 = time.perf_counter()

    q_emb = embed_query_ollama(search_q)

    latency["embed_s"] = round(
        time.perf_counter() - t0, 3
    )

    # Step 3: Vector retrieval
    t0 = time.perf_counter()

    store = get_vector_store(
        backend=backend,
        collection_name=collection,
        dim=len(q_emb),
    )

    fetch_k = (
        topk * FETCH_MULTIPLIER
        if mmr else topk
    )

    results = store.query(q_emb, k=fetch_k)

    if mmr and len(results) > topk:

        candidate_embeddings = embed_ollama(
            [r["document"] for r in results]
        )

        order = mmr_rerank(
            q_emb,
            candidate_embeddings,
            topk,
        )

        results = [results[i] for i in order]

    else:
        results = results[:topk]

    latency["retrieve_s"] = round(
        time.perf_counter() - t0, 3
    )

    # Step 4: Build the RAG prompt
    prompt = build_prompt(question, results)

    return {
        "prompt": prompt,
        "results": results,
        "query_used": search_q,
        "latency": latency,
    }


# ------------------------------------------------------------
# NON-STREAMING GENERATION
# ------------------------------------------------------------

def _generate(prompt: str, llm: str) -> tuple:

    t0 = time.perf_counter()

    if llm == "gemini":

        text = gemini_text(prompt)

    elif llm == "ollama":

        response = ollama.chat(
            model=OLLAMA_CHAT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        text = (
            response["message"].get("content") or ""
        ).strip()

    else:
        raise ValueError(
            f"Unknown LLM: {llm}. Choose gemini or ollama."
        )

    elapsed = time.perf_counter() - t0

    return text, elapsed


# ------------------------------------------------------------
# STREAMING GENERATION
# ------------------------------------------------------------

def _generate_stream(
    prompt: str,
    llm: str,
    max_retries: int = 5,
) -> Generator[str, None, None]:

    if llm == "ollama":

        stream = ollama.chat(
            model=OLLAMA_CHAT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            stream=True,
        )

        for chunk in stream:

            piece = (
                chunk["message"].get("content") or ""
            )

            if piece:
                yield piece

        return

    if llm != "gemini":
        raise ValueError(
            f"Unknown LLM: {llm}"
        )

    for attempt in range(max_retries):

        yielded_any = False

        try:

            stream = (
                get_gemini_client()
                .models.generate_content_stream(
                    model=GEMINI_CHAT_MODEL,
                    contents=prompt,
                )
            )

            for chunk in stream:

                if chunk.text:
                    yielded_any = True
                    yield chunk.text

            return

        except Exception as error:

            is_rate_limit = (
                "429" in str(error)
                or "RESOURCE_EXHAUSTED"
                in str(error).upper()
            )

            if (
                not is_rate_limit
                or yielded_any
                or attempt == max_retries - 1
            ):
                raise

            wait = (
                2 ** attempt
                + random.uniform(0, 1)
            )

            print(
                f"\n[backoff] Rate limit. "
                f"Waiting {wait:.1f}s..."
            )

            time.sleep(wait)


# ------------------------------------------------------------
# CITATION EXTRACTION
# ------------------------------------------------------------

def _citations(
    text: str,
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    markers = sorted({
        int(marker)
        for marker in CITE_RE.findall(text)
    })

    citations = []

    for number in markers:

        if 1 <= number <= len(results):

            result = results[number - 1]

            metadata = result.get(
                "metadata", {}
            ) or {}

            citations.append({
                "marker": number,
                "source": metadata.get(
                    "source", "unknown"
                ),
                "page": metadata.get(
                    "page", "?"
                ),
                "score": round(
                    result.get("score", 0.0),
                    3,
                ),
            })

    return citations


# ------------------------------------------------------------
# ABSTENTION DETECTION
# ------------------------------------------------------------

def _abstained(text: str) -> bool:

    lowered = text.lower()

    return any(
        marker in lowered
        for marker in ABSTAIN_MARKERS
    )


# ------------------------------------------------------------
# PUBLIC RAG FUNCTION
# USED BY DAY4 AGENT
# ------------------------------------------------------------

def answer(
    question: str,
    backend: str = "chroma",
    llm: str = "gemini",
    topk: int = 5,
    mmr: bool = False,
    rewrite: bool = False,
    collection: str = DEFAULT_COLLECTION,
    min_score: Optional[float] = None,
) -> Dict[str, Any]:

    prep = _prepare(
        question=question,
        backend=backend,
        topk=topk,
        mmr=mmr,
        rewrite=rewrite,
        collection=collection,
    )

    results = prep["results"]

    # Optional retrieval-score guardrail
    if min_score is not None:

        if (
            not results
            or results[0].get("score", 0.0)
            < min_score
        ):

            latency = dict(prep["latency"])

            latency["generate_s"] = 0.0

            latency["total_s"] = round(
                sum(latency.values()), 3
            )

            return {
                "text": (
                    "I don't know — the answer is "
                    "not in the knowledge base."
                ),
                "citations": [],
                "scores": [
                    round(r.get("score", 0.0), 3)
                    for r in results
                ],
                "abstained": True,
                "latency": latency,
                "retrieved": results,
                "query_used": prep["query_used"],
            }

    # Generate answer
    text, generation_seconds = _generate(
        prep["prompt"],
        llm,
    )

    latency = dict(prep["latency"])

    latency["generate_s"] = round(
        generation_seconds, 3
    )

    latency["total_s"] = round(
        sum(latency.values()), 3
    )

    return {
        "text": text,
        "citations": _citations(text, results),
        "scores": [
            round(r.get("score", 0.0), 3)
            for r in results
        ],
        "abstained": _abstained(text),
        "latency": latency,
        "retrieved": results,
        "query_used": prep["query_used"],
    }


# ------------------------------------------------------------
# RESULT REPORT
# ------------------------------------------------------------

def _report(
    result_text: str,
    results: List[Dict],
    citations: List[Dict],
    latency: Dict[str, float],
    abstained: bool,
) -> None:

    if citations:

        print("\n[citations]")

        for citation in citations:

            print(
                f"  [{citation['marker']}] "
                f"{citation['source']} "
                f"p.{citation['page']} "
                f"(score {citation['score']})"
            )

    else:

        print(
            "[citations] None"
        )

    print(
        f"[abstained] "
        f"{'yes' if abstained else 'no'}"
    )

    parts = " | ".join(
        f"{stage}={value:.3f}s"
        for stage, value in latency.items()
    )

    print(f"[latency] {parts}")


# ------------------------------------------------------------
# SINGLE QUESTION EXECUTION
# ------------------------------------------------------------

def _run_once(
    question: str,
    args: argparse.Namespace,
) -> None:

    prep = _prepare(
        question=question,
        backend=args.backend,
        topk=args.topk,
        mmr=(args.mmr == "on"),
        rewrite=(args.rewrite == "on"),
        collection=args.collection,
    )

    results = prep["results"]

    if args.stream == "on":

        print(
            "[answer] ",
            end="",
            flush=True,
        )

        parts = []

        t0 = time.perf_counter()

        for token in _generate_stream(
            prep["prompt"],
            args.llm,
        ):

            print(
                token,
                end="",
                flush=True,
            )

            parts.append(token)

        generation_seconds = (
            time.perf_counter() - t0
        )

        text = "".join(parts)

        print()

    else:

        text, generation_seconds = _generate(
            prep["prompt"],
            args.llm,
        )

        print(f"[answer] {text}")

    latency = dict(prep["latency"])

    latency["generate_s"] = round(
        generation_seconds, 3
    )

    latency["total_s"] = round(
        sum(latency.values()), 3
    )

    _report(
        text,
        results,
        _citations(text, results),
        latency,
        _abstained(text),
    )


# ------------------------------------------------------------
# COMMAND LINE INTERFACE
# ------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Day 3 RAG Engine - "
            "Citations, Retrieval, and Generation"
        )
    )

    parser.add_argument(
        "--question",
        default=None,
    )

    parser.add_argument(
        "--backend",
        default="chroma",
        choices=[
            "chroma",
            "faiss",
            "pinecone",
        ],
    )

    parser.add_argument(
        "--llm",
        default="gemini",
        choices=[
            "gemini",
            "ollama",
        ],
    )

    parser.add_argument(
        "--topk",
        default=5,
        type=int,
    )

    parser.add_argument(
        "--mmr",
        default="off",
        choices=["on", "off"],
    )

    parser.add_argument(
        "--rewrite",
        default="off",
        choices=["on", "off"],
    )

    parser.add_argument(
        "--stream",
        default="on",
        choices=["on", "off"],
    )

    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
    )

    args = parser.parse_args()

    if args.question:

        _run_once(
            args.question,
            args,
        )

        return

    print("=" * 60)
    print("DAY 3 RAG ENGINE")
    print("=" * 60)

    print(
        f"Backend: {args.backend}"
    )

    print(
        f"LLM: {args.llm}"
    )

    print(
        "Type a question or 'quit' to exit."
    )

    while True:

        question = input(
            "\nYou: "
        ).strip()

        if question.lower() in (
            "quit",
            "exit",
            "q",
        ):

            print("Goodbye!")
            break

        if not question:
            continue

        try:

            _run_once(
                question,
                args,
            )

        except Exception as error:

            print(
                f"[error] "
                f"{type(error).__name__}: "
                f"{error}"
            )


# ------------------------------------------------------------
# PROGRAM ENTRY POINT
# ------------------------------------------------------------

if __name__ == "__main__":
    main()