import tiktoken
import re
from typing import List, Dict

enc = tiktoken.get_encoding("cl100k_base")


# -----------------------------
# Token utilities
# -----------------------------
def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def trim_to_token_limit(text: str, max_tokens: int) -> str:
    tokens = enc.encode(text)
    return enc.decode(tokens[:max_tokens])


def get_last_tokens(text: str, n: int) -> str:
    """
    Safe token-based overlap extraction
    """
    tokens = enc.encode(text)
    return enc.decode(tokens[-n:]) if len(tokens) > n else text


# -----------------------------
# Step 1: structure splitting
# -----------------------------
def split_into_sections(text: str) -> List[str]:
    """
    Split markdown by headings first (structure-aware)
    """
    text = text.replace("\r\n", "\n").replace("\n\n", "\n\n")
    sections = re.split(r"\n(?=# )|\n(?=## )|\n(?=### )", text)
    return [s.strip() for s in sections if s.strip()]


def split_paragraphs(section: str) -> List[str]:
    return [p.strip() for p in section.split("\n\n") if p.strip()]


# -----------------------------
# Step 2: improved chunk builder
# -----------------------------
def create_chunks_from_doc(
    doc: Dict,
    max_tokens: int = 300,
    overlap_tokens: int = 40,
) -> List[Dict]:
    """
    Clean hybrid chunking:
    - structure-aware (sections + paragraphs)
    - stable token limits
    - safe overlap (no decode corruption)
    """

    text = doc["text"]
    source = doc["source"]

    sections = split_into_sections(text)

    chunks = []
    chunk_index = 0

    current_text = ""
    current_tokens = 0

    def flush_chunk():
        nonlocal chunk_index, current_text, current_tokens

        if not current_text.strip():
            return

        cleaned = current_text.strip()
        cleaned = trim_to_token_limit(cleaned, max_tokens)

        chunks.append(
            {
                "chunk_id": f"{source}_{chunk_index}",
                "source": source,
                "text": cleaned,
                "chunk_index": chunk_index,
            }
        )

        chunk_index += 1

        # SAFE overlap (based on full text, not decoded rebuild)
        overlap = get_last_tokens(cleaned, overlap_tokens)

        current_text = overlap
        current_tokens = count_tokens(overlap)

    for section in sections:
        paragraphs = split_paragraphs(section)

        for para in paragraphs:
            para_tokens = count_tokens(para)

            if current_tokens + para_tokens > max_tokens:
                flush_chunk()

            if current_text:
                current_text += "\n\n" + para
            else:
                current_text = para

            current_tokens = count_tokens(current_text)

    flush_chunk()

    return chunks
