DEFAULT_REVIEW_MODEL = "google/flan-t5-small"


def review_chunk_with_small_llm(text: str, model_name: str = DEFAULT_REVIEW_MODEL) -> str:
    """Ask a small Hugging Face model if a chunk looks complete.

    This is optional. The rule-based validator is the default because it is fast
    and easy to audit; this helper is useful for manually reviewing doubtful
    chunks during experimentation.
    """
    try:
        from transformers import pipeline
    except ImportError as error:
        raise ImportError("Install transformers to review chunks with a small LLM.") from error

    reviewer = pipeline("text2text-generation", model=model_name)
    prompt = (
        "Answer only OK or REVIEW. Does this Spanish text end with a complete idea?\n\n"
        f"{text[:1800]}"
    )
    result = reviewer(prompt, max_new_tokens=8, do_sample=False)
    return result[0]["generated_text"].strip()
