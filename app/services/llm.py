import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"


# ----------------------------
# Normalize context safely
# ----------------------------
def normalize_context(c):
    """
    Handles:
    - dict context
    - Qdrant ScoredPoint
    """

    if isinstance(c, dict):
        return c

    if hasattr(c, "payload"):
        return {
            "source": c.payload.get("source", "unknown"),
            "text": c.payload.get("text", ""),
        }

    return {
        "source": "unknown",
        "text": str(c),
    }


# ----------------------------
# Confidence scoring (simple but effective)
# ----------------------------
def compute_confidence(contexts: list[dict]) -> float:
    """
    Heuristic confidence:
    - more chunks = better
    - assumes upstream reranker already improved ordering
    """

    if not contexts:
        return 0.0

    # signal 1: quantity
    quantity_score = min(len(contexts) / 5, 1.0)

    # signal 2: basic content richness
    text_lengths = [len(c.get("text", "")) for c in contexts]
    avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
    length_score = min(avg_length / 500, 1.0)

    confidence = (0.6 * quantity_score) + (0.4 * length_score)

    return round(confidence, 3)


# ----------------------------
# Main generator (STRICT MODE)
# ----------------------------
def generate_answer(query: str, contexts: list[dict]) -> dict:
    """
    STRICT ENTERPRISE RAG MODE

    - No hallucination allowed
    - Must use only context
    - Refuse if weak retrieval
    """

    normalized = [normalize_context(c) for c in contexts]

    # ----------------------------
    # Build context block
    # ----------------------------
    context_text = "\n\n".join(
        f"[{i}] SOURCE: {c['source']}\nTEXT: {c['text']}"
        for i, c in enumerate(normalized)
    )

    confidence = compute_confidence(normalized)

    # ----------------------------
    # STRICT REFUSAL LOGIC
    # ----------------------------
    if confidence < 0.35 or len(normalized) == 0:
        return {
            "answer": "I don't know based on the provided documents.",
            "confidence": confidence,
            "used_sources": [],
        }

    # ----------------------------
    # SYSTEM PROMPT (STRICT MODE)
    # ----------------------------
    system_prompt = """
You are a STRICT enterprise RAG assistant.

RULES:
- Use ONLY the provided context.
- NEVER use outside knowledge.
- If answer is not in context, say "I don't know based on the provided documents."
- Every factual sentence MUST include citation like [0], [1].
- Do NOT guess or infer missing information.
- Be concise and grounded.
"""

    user_prompt = f"""
QUESTION:
{query}

CONTEXT:
{context_text}

INSTRUCTION:
Answer ONLY from the context. Always include citations.
"""

    # ----------------------------
    # LLM CALL
    # ----------------------------
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "confidence": confidence,
        "used_sources": [c["source"] for c in normalized],
    }
