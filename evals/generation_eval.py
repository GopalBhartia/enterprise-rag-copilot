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
# Simple eval helpers
# ----------------------------
def contains_citation(answer):
    return "[" in answer and "]" in answer


def source_coverage(retrieved, expected):
    return any(r in expected for r in retrieved)


# ----------------------------
# Eval
# ----------------------------
def run_eval():
    print("\n===== GENERATION EVAL (WITH LLM) =====\n")

    total = len(TEST_SET)

    citation_ok = 0
    grounded_ok = 0

    for item in TEST_SET:
        query = item["query"]
        expected = item["expected"]

        query_vector = get_embedding(query)
        results = search(query_vector, top_k=TOP_K)

        retrieved = [r.payload["source"] for r in results]

        contexts = [
            {
                "source": r.payload["source"],
                "text": r.payload["text"],
            }
            for r in results
        ]

        response = generate_answer(query, contexts)
        answer = response["answer"]

        cit = contains_citation(answer)
        grounded = source_coverage(retrieved, expected)

        citation_ok += int(cit)
        grounded_ok += int(grounded)

        print("\n-----------------------------")
        print("QUERY:", query)
        print("RETRIEVED:", retrieved)
        print("CITATIONS OK:", cit)
        print("GROUNDED RETRIEVAL:", grounded)
        print("ANSWER:\n", answer)

    print("\n===== SUMMARY =====")
    print(f"Citation Rate: {citation_ok / total:.2f}")
    print(f"Grounded Retrieval Rate: {grounded_ok / total:.2f}")


if __name__ == "__main__":
    run_eval()
