import numpy as np


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Load a small multilingual Hugging Face embedding model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ImportError("Install sentence-transformers to create embeddings.") from error

    return SentenceTransformer(model_name)


def create_embeddings(texts: list[str], model) -> np.ndarray:
    """Create normalized embeddings for cosine similarity."""
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings.astype("float32")
