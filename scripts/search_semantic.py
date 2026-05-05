import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag.embeddings import DEFAULT_EMBEDDING_MODEL, create_embeddings, load_embedding_model
from rag.vector_store import load_faiss_index, search_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test semantic search over the week 1 index.")
    parser.add_argument("query", help="Question or search query.")
    parser.add_argument("--chunks-path", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--index-path", type=Path, default=Path("indexes/faiss.index"))
    parser.add_argument("--model-name", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = _load_chunks(args.chunks_path)
    index = load_faiss_index(args.index_path)
    model = load_embedding_model(args.model_name)
    query_embedding = create_embeddings([args.query], model)
    scores, indexes = search_index(index, query_embedding, args.top_k)

    for rank, (score, chunk_index) in enumerate(zip(scores, indexes), start=1):
        if chunk_index == -1:
            continue

        chunk = chunks[chunk_index]
        preview = chunk["text"].replace("\n", " ")[:450]
        print(f"\n#{rank} score={score:.4f} id={chunk['id']} source={chunk['source']}")
        print(preview)


def _load_chunks(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


if __name__ == "__main__":
    main()
