import json
from pathlib import Path

from app.services.vector_store import upsert_chunks

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = Path("data/raw")
MARKER_FILE = Path("data/.ingested")


# -----------------------------
# Main ingestion function
# -----------------------------
def run_ingestion_if_needed():
    """
    Automatically runs ingestion ONLY ONCE.

    Uses a marker file:
    - If data/.ingested exists → skip ingestion
    - Otherwise → ingest + create marker
    """

    # -----------------------------
    # 1. CHECK IF ALREADY INGESTED
    # -----------------------------
    if MARKER_FILE.exists():
        print("✅ Ingestion already completed. Skipping.")
        return

    print("🚀 Starting automatic ingestion...")

    # -----------------------------
    # 2. LOAD DATA
    # -----------------------------
    all_chunks = []

    if not DATA_PATH.exists():
        print(f"⚠️ Data folder not found: {DATA_PATH}")
        return

    for file in DATA_PATH.glob("**/*.json"):
        try:
            with open(file, "r") as f:
                data = json.load(f)

                if isinstance(data, list):
                    all_chunks.extend(data)
                else:
                    print(f"⚠️ Skipping invalid format in {file}")

        except Exception as e:
            print(f"❌ Failed to read {file}: {e}")

    # -----------------------------
    # 3. VALIDATION
    # -----------------------------
    if not all_chunks:
        print("⚠️ No chunks found for ingestion. Exiting.")
        return

    print(f"📦 Total chunks to ingest: {len(all_chunks)}")

    # -----------------------------
    # 4. UPSERT INTO VECTOR DB
    # -----------------------------
    try:
        upsert_chunks(all_chunks)
        print("✅ Vector DB ingestion successful")

    except Exception as e:
        print(f"❌ Vector DB ingestion failed: {e}")
        return

    # -----------------------------
    # 5. CREATE MARKER FILE
    # -----------------------------
    try:
        MARKER_FILE.parent.mkdir(parents=True, exist_ok=True)
        MARKER_FILE.write_text("ingested=true\n")

        print("🧾 Ingestion marker created at data/.ingested")

    except Exception as e:
        print(f"❌ Failed to create marker file: {e}")
        return

    print("🎉 Auto ingestion completed successfully.")
