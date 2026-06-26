import re

from app.services.embeddings import get_embedding
from app.services.vector_store import search
from app.services.llm import generate_answer


TEST_SET = [
    {"query": "what is fastapi", "expected": ["fastapi/overview.md"]},
    {
        "query": "what is dependency injection in fastapi",
        "expected": ["fastapi/dependencies.md"],
    },
    {
        "query": "how does fastapi handle startup and shutdown",
        "expected": ["fastapi/lifecycle.md"],
    },
    {"query": "what is pydantic validation", "expected": ["pydantic/validations.md"]},
    {"query": "what is docker container", "expected": ["docker/overview.md"]},
]

TOP_K = 5


# ----------------------------
# Helpers
# ----------------------------


def extract_sources(results):
    return [r.payload["source"] for r in results]


def extract_citations(answer: str):
    """
    Extract citation indices like [0], [1], [2]
    """
    return [int(x) for x in re.findall(r"\[(\d+)\]", answer)]


def is_grounded(answer, contexts):
    """
    Grounded if:
    - model uses at least one citation index
    - AND all citation indices exist in context range
    """
    citations = extract_citations(answer)

    if not citations:
        return False

    max_index = len(contexts) - 1

    return all(0 <= c <= max_index for c in citations)


def has_refusal(answer):
    return "I don't know" in answer.lower()


def has_hallucination(answer, contexts):
    """
    Simple hallucination signal:
    - citations exist
    - but referenced content does not align with context sources
    """
    citations = extract_citations(answer)

    if not citations:
        return True

    for c in citations:
        if c >= len(contexts):
            return True

    return False


# ----------------------------
# Evaluation
# ----------------------------


def run_eval():
    print("\n===== ADVANCED EVALUATION =====\n")

    total = len(TEST_SET)

    grounded_count = 0
    refusal_count = 0
    hallucination_count = 0

    for item in TEST_SET:
        query = item["query"]

        query_vector = get_embedding(query)
        results = search(query_vector, top_k=TOP_K)

        retrieved = extract_sources(results)

        contexts = [
            {
                "source": r.payload["source"],
                "text": r.payload["text"],
            }
            for r in results
        ]

        response = generate_answer(query, contexts)
        answer = response["answer"]

        grounded = is_grounded(answer, contexts)
        refusal = has_refusal(answer)
        hallucination = has_hallucination(answer, contexts)

        grounded_count += int(grounded)
        refusal_count += int(refusal)
        hallucination_count += int(hallucination)

        print("\n-----------------------------")
        print("QUERY:", query)
        print("RETRIEVED:", retrieved)
        print("GROUNDED:", grounded)
        print("REFUSAL:", refusal)
        print("HALLUCINATION:", hallucination)
        print("ANSWER:\n", answer)

    print("\n===== SUMMARY =====")
    print(f"Grounded Rate: {grounded_count / total:.2f}")
    print(f"Refusal Rate: {refusal_count / total:.2f}")
    print(f"Hallucination Rate: {hallucination_count / total:.2f}")


if __name__ == "__main__":
    run_eval()
