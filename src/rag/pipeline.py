from pathlib import Path

from rag.embeddings import DEFAULT_EMBEDDING_MODEL, load_embedding_model
from rag.generator import DEFAULT_GENERATION_MODEL, FALLBACK_ANSWER, generate_answer, load_generation_model
from rag.retriever import format_context, load_chunks, retrieve_context
from rag.vector_store import load_faiss_index


class RagChatbot:
    """End-to-end RAG pipeline: retrieval plus Qwen answer generation."""

    def __init__(
        self,
        chunks_path: Path = Path("data/processed/chunks.jsonl"),
        index_path: Path = Path("indexes/faiss.index"),
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        generation_model_name: str = DEFAULT_GENERATION_MODEL,
        local_files_only: bool = True,
    ) -> None:
        # Load retrieval resources once; Qwen is loaded lazily on the first real answer.
        self.chunks = load_chunks(chunks_path)
        self.index = load_faiss_index(index_path)
        self.embedding_model = load_embedding_model(embedding_model_name, local_files_only=local_files_only)
        self.generation_model_name = generation_model_name
        self.local_files_only = local_files_only
        self.tokenizer = None
        self.generation_model = None

    def ask(
        self,
        query: str,
        top_k: int = 4,
        min_score: float = 0.25,
        max_context_chars: int = 4000,
        max_new_tokens: int = 220,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> dict:
        # Retrieval happens before generation so the model only sees project evidence.
        results = retrieve_context(query, self.chunks, self.index, self.embedding_model, top_k, min_score)
        context = format_context(results, max_context_chars)
        if not context:
            return {"answer": FALLBACK_ANSWER, "context": results}

        self._load_generator()
        answer = generate_answer(
            query,
            context,
            self.tokenizer,
            self.generation_model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return {"answer": answer, "context": results}

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        min_score: float = 0.25,
    ) -> list[dict]:
        return retrieve_context(query, self.chunks, self.index, self.embedding_model, top_k, min_score)

    def _load_generator(self) -> None:
        if self.tokenizer is not None and self.generation_model is not None:
            return

        self.tokenizer, self.generation_model = load_generation_model(
            self.generation_model_name,
            local_files_only=self.local_files_only,
        )
