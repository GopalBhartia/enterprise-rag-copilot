import json
from datetime import datetime
from pathlib import Path

FEEDBACK_FILE = Path("logs/feedback.jsonl")


def save_feedback(feedback: dict) -> None:
    """
    Save one feedback record as JSONL.
    """
    FEEDBACK_FILE.parent.mkdir(exist_ok=True)

    feedback["timestamp"] = datetime.utcnow().isoformat()

    with FEEDBACK_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(feedback) + "\n")
