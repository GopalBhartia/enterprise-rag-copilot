import json
from pathlib import Path

import requests

API_URL = "http://localhost:8000/rag/ask"

DATASET_PATH = Path("evals/datasets/golden_dataset.json")
OUTPUT_PATH = Path("evals/results.json")


def load_dataset():
    with open(DATASET_PATH, "r") as f:
        return json.load(f)


def evaluate():
    dataset = load_dataset()

    results = []

    print(f"Running evaluation on {len(dataset)} questions...\n")

    for item in dataset:
        response = requests.post(
            API_URL,
            json={
                "query": item["question"],
            },
            timeout=120,
        )

        if response.status_code != 200:
            print(f"Failed: {item['question']}")
            continue

        prediction = response.json()

        result = {
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "expected_source": item["expected_source"],
            "generated_answer": prediction["answer"],
            "citations": prediction["citations"],
            "contexts": prediction["contexts"],
            "trace_id": prediction["trace_id"],
        }

        results.append(result)

        print(f"✓ {item['question']}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved evaluation results to {OUTPUT_PATH}")


if __name__ == "__main__":
    evaluate()
