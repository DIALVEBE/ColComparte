import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag.chunker import ChunkingConfig, build_chunks, chunk_to_record
from rag.document_loader import load_documents
from rag.embeddings import DEFAULT_EMBEDDING_MODEL, create_embeddings, load_embedding_model
from rag.vector_store import save_faiss_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build week 1 RAG chunks and embeddings.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--chunks-path", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--embeddings-path", type=Path, default=Path("data/processed/embeddings.npy"))
    parser.add_argument("--index-path", type=Path, default=Path("indexes/faiss.index"))
    parser.add_argument("--model-name", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--min-words", type=int, default=35)
    parser.add_argument("--target-words", type=int, default=140)
    parser.add_argument("--max-words", type=int, default=240)
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--download-model", action="store_true", help="Allow downloading the embedding model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = load_documents(args.input_dir)

    if not documents:
        raise SystemExit(f"No supported documents found in {args.input_dir}.")

    config = ChunkingConfig(
        min_words=args.min_words,
        target_words=args.target_words,
        max_words=args.max_words,
    )

    records = []
    for document in documents:
        # Build clean chunks per source, then assign global ids for FAISS alignment.
        chunks = build_chunks(document["text"], document["source"], config)
        for chunk in chunks:
            record = chunk_to_record(chunk)
            record["id"] = f"chunk-{len(records) + 1:04d}"
            records.append(record)

    args.chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with args.chunks_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.skip_embeddings:
        reviewed = sum(record["validation_status"] == "review" for record in records)
        print(f"Documents processed: {len(documents)}")
        print(f"Chunks created: {len(records)}")
        print(f"Chunks to review: {reviewed}")
        print(f"Chunks saved to: {args.chunks_path}")
        print("Embeddings skipped.")
        return

    model = load_embedding_model(args.model_name, local_files_only=not args.download_model)
    # The vector index position must match the order of records in chunks.jsonl.
    embeddings = create_embeddings([record["text"] for record in records], model)
    args.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.embeddings_path, embeddings)
    save_faiss_index(embeddings, args.index_path)

    reviewed = sum(record["validation_status"] == "review" for record in records)
    print(f"Documents processed: {len(documents)}")
    print(f"Chunks created: {len(records)}")
    print(f"Chunks to review: {reviewed}")
    print(f"Chunks saved to: {args.chunks_path}")
    print(f"FAISS index saved to: {args.index_path}")


if __name__ == "__main__":
    main()
