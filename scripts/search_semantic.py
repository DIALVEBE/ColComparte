import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag.embeddings import DEFAULT_EMBEDDING_MODEL, create_embeddings, load_embedding_model
from rag.retriever import load_chunks
from rag.vector_store import load_faiss_index, search_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test semantic search over the week 1 index.")
    parser.add_argument("query", help="Question or search query.")
    parser.add_argument("--chunks-path", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--index-path", type=Path, default=Path("indexes/faiss.index"))
    parser.add_argument("--model-name", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--download-model", action="store_true", help="Allow downloading the embedding model.")
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=0,
        help="Limit printed chunk text to this many characters. 0 prints the full chunk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = load_chunks(args.chunks_path)
    index = load_faiss_index(args.index_path)
    model = load_embedding_model(args.model_name, local_files_only=not args.download_model)
    query_embedding = create_embeddings([args.query], model)
    scores, indexes = search_index(index, query_embedding, args.top_k)

    # indexes are FAISS row positions; the same positions are used in chunks.jsonl.
    for rank, (score, chunk_index) in enumerate(zip(scores, indexes), start=1):
        if chunk_index == -1:
            continue

        chunk = chunks[chunk_index]
        preview = _format_preview(chunk["text"], args.preview_chars)
        print(f"\n#{rank} score={score:.4f} id={chunk['id']} source={chunk['source']}")
        print(preview)


def _format_preview(text: str, preview_chars: int) -> str:
    text = text.strip()
    if preview_chars <= 0 or len(text) <= preview_chars:
        return text

    preview = text[:preview_chars].rstrip()
    last_space = preview.rfind(" ")
    if last_space > int(preview_chars * 0.75):
        preview = preview[:last_space]
    return f"{preview}..."


if __name__ == "__main__":
    main()
