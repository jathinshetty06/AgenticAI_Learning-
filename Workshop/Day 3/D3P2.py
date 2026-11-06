
from typing import List, Dict, Any

RAG_PROMPT_TEMPLATE = """You are a precise knowledge-base assistant answering questions
about a campus handbook. Follow the rules below without exception.

Context blocks:
{context}

Rules:
1. Answer ONLY using facts that appear in the context blocks above.
   Do not use prior knowledge, even if you know the answer.
2. After each factual claim, cite the supporting block number in square brackets,
   e.g. [1] or [2][3]. Every claim needs at least one citation.
3. If the context blocks do not contain the information needed, reply with exactly:
   "I don't know — the answer is not in the knowledge base."
   Do not guess, do not blend in outside facts.
4. Keep the answer to 2-6 sentences unless the question demands more detail.

Question: {question}

Answer:"""


def build_prompt(
    question: str,
    results: List[Dict[str, Any]]
) -> str:

    blocks = []

    for i, result in enumerate(results, 1):

        metadata = result.get("metadata") or {}

        source = metadata.get("source", "unknown")
        page = metadata.get("page", "?")
        document = result.get("document", "")

        blocks.append(
            f"[{i}] (source: {source}, page: {page})\n{document}"
        )

    context_str = (
        "\n\n".join(blocks)
        if blocks
        else "(no context blocks retrieved)"
    )

    return RAG_PROMPT_TEMPLATE.format(
        context=context_str,
        question=question
    )


if __name__ == "__main__":
    sample_results = [
        {
            "document": "Library opens at 9 AM.",
            "metadata": {"source": "handbook.pdf", "page": 5}
        }
    ]

    print(build_prompt("When does the library open?", sample_results))
