import os
from typing import List

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


EMBEDDING_MODEL = "text-embedding-3-small"


def get_embedding(text: str) -> List[float]:
    """
    Get embedding for a single text chunk.
    """

    text = text.replace("\n", " ")

    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)

    return response.data[0].embedding


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Batch embedding (important for performance later).
    """

    response = client.embeddings.create(
        model=EMBEDDING_MODEL, input=[t.replace("\n", " ") for t in texts]
    )

    return [item.embedding for item in response.data]
