import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag.embeddings import DEFAULT_EMBEDDING_MODEL
from rag.generator import DEFAULT_GENERATION_MODEL
from rag.pipeline import RagChatbot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask the Colombia Comparte RAG chatbot.")
    parser.add_argument("query", help="Question for the chatbot.")
    parser.add_argument("--chunks-path", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--index-path", type=Path, default=Path("indexes/faiss.index"))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--generation-model", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-score", type=float, default=0.25)
    parser.add_argument("--max-context-chars", type=int, default=4000)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--show-context", action="store_true")
    parser.add_argument("--no-generate", action="store_true", help="Only retrieve and print context.")
    parser.add_argument("--download-models", action="store_true", help="Allow downloading Hugging Face models.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # CLI version of the same pipeline used by Streamlit, useful for quick tests.
    local_files_only = not args.download_models
    chatbot = RagChatbot(
        chunks_path=args.chunks_path,
        index_path=args.index_path,
        embedding_model_name=args.embedding_model,
        generation_model_name=args.generation_model,
        local_files_only=local_files_only,
    )
    results = chatbot.retrieve(args.query, args.top_k, args.min_score)

    if args.show_context or args.no_generate:
        _print_context(results)

    if args.no_generate:
        return

    response = chatbot.ask(
        args.query,
        top_k=args.top_k,
        min_score=args.min_score,
        max_context_chars=args.max_context_chars,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(response["answer"])


def _print_context(results: list[dict]) -> None:
    if not results:
        print("No retrieved context passed the score threshold.")
        return

    for rank, result in enumerate(results, start=1):
        print(f"\n#{rank} score={result['score']:.4f} id={result['id']} source={result['source']}")
        print(result["text"])


if __name__ == "__main__":
    main()
