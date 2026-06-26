from pathlib import Path

from app.services.chunking import create_chunks_from_doc
from app.services.embeddings import get_embeddings
from app.services.vector_store import (
    create_collection,
    upsert_chunks,
)

from app.services.bm25 import BM25Retriever


# ----------------------------
# File loaders
# ----------------------------


def load_pdf(file_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    return text


def load_text(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def load_file(file_path: Path) -> str:
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(file_path)
    elif suffix in [".md", ".txt"]:
        return load_text(file_path)
    else:
        return ""


# ----------------------------
# Ingestion pipeline
# ----------------------------


def ingest_folder(folder: str):
    folder_path = Path(folder)

    all_chunks = []

    for file in folder_path.rglob("*"):
        if not file.is_file():
            continue

        print(f"Processing: {file.relative_to(folder_path)}")

        text = load_file(file)

        if not text.strip():
            continue

        doc = {"source": str(file.relative_to(folder_path)), "text": text}

        chunks = create_chunks_from_doc(doc)
        all_chunks.extend(chunks)

    return all_chunks


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    print("Starting ingestion pipeline...")

    chunks = ingest_folder("data/raw")

    if not chunks:
        print("No chunks generated.")
        exit(0)

    print(f"Generated {len(chunks)} chunks")

    # ----------------------------
    # embeddings
    # ----------------------------

    texts = [c["text"] for c in chunks]

    print("Generating embeddings...")
    embeddings = get_embeddings(texts)

    # ----------------------------
    # vector DB
    # ----------------------------

    print("Initializing vector DB...")
    create_collection(vector_size=len(embeddings[0]))

    print("Upserting chunks...")
    upsert_chunks(chunks, embeddings)

    # ----------------------------
    # BM25 INDEX BUILD
    # ----------------------------

    print("Building BM25 index...")
    bm25 = BM25Retriever()
    bm25.build(chunks)

    print("Ingestion complete (vector + BM25 ready)")
