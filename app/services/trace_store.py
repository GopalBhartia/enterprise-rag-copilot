import json
from datetime import datetime
from pathlib import Path

TRACE_FILE = Path("data/traces.jsonl")


def write_trace(trace: dict):
    """
    Append a single structured RAG trace to JSONL storage.
    """

    TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)

    trace_record = {
        "timestamp": datetime.utcnow().isoformat(),
        **trace,
    }

    with open(TRACE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace_record) + "\n")
