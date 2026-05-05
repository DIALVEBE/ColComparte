import re
from dataclasses import dataclass

from rag.text_cleaner import clean_text, split_paragraphs


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    start_paragraph: int
    end_paragraph: int
    word_count: int
    validation_status: str
    validation_notes: list[str]


@dataclass
class ChunkingConfig:
    min_words: int = 180
    target_words: int = 320
    max_words: int = 500
    overlap_sentences: int = 1


def build_chunks(text: str, source: str, config: ChunkingConfig | None = None) -> list[Chunk]:
    """Create paragraph-aware chunks that avoid cutting ideas in half."""
    config = config or ChunkingConfig()
    paragraphs = split_paragraphs(clean_text(text))
    chunks: list[Chunk] = []
    current: list[str] = []
    start_index = 0

    for paragraph_index, paragraph in enumerate(paragraphs):
        paragraph_words = _count_words(paragraph)
        current_words = _count_words(" ".join(current))

        if current and current_words + paragraph_words > config.max_words:
            chunks.append(_create_chunk(chunks, current, source, start_index, paragraph_index - 1, config))
            current = _build_overlap(current, config.overlap_sentences)
            start_index = max(start_index, paragraph_index - len(current))

        if paragraph_words > config.max_words:
            # Very long paragraphs are split by sentences as a last resort.
            sentence_chunks = _split_long_paragraph(paragraph, config.max_words)
            for sentence_chunk in sentence_chunks:
                candidate = current + [sentence_chunk]
                if _count_words(" ".join(candidate)) >= config.min_words:
                    chunks.append(_create_chunk(chunks, candidate, source, start_index, paragraph_index, config))
                    current = _build_overlap(candidate, config.overlap_sentences)
                    start_index = paragraph_index
                else:
                    current = candidate
            continue

        current.append(paragraph)

        if _count_words(" ".join(current)) >= config.target_words:
            chunks.append(_create_chunk(chunks, current, source, start_index, paragraph_index, config))
            current = _build_overlap(current, config.overlap_sentences)
            start_index = paragraph_index if current else paragraph_index + 1

    if current:
        if chunks and _count_words(" ".join(current)) < config.min_words:
            chunks[-1] = _merge_last_chunk(chunks[-1], current, config)
        else:
            chunks.append(_create_chunk(chunks, current, source, start_index, len(paragraphs) - 1, config))

    return chunks


def chunk_to_record(chunk: Chunk) -> dict:
    """Convert a chunk to a JSON-serializable dictionary."""
    return {
        "id": chunk.id,
        "text": chunk.text,
        "source": chunk.source,
        "start_paragraph": chunk.start_paragraph,
        "end_paragraph": chunk.end_paragraph,
        "word_count": chunk.word_count,
        "validation_status": chunk.validation_status,
        "validation_notes": chunk.validation_notes,
    }


def _create_chunk(
    chunks: list[Chunk],
    paragraphs: list[str],
    source: str,
    start_paragraph: int,
    end_paragraph: int,
    config: ChunkingConfig,
) -> Chunk:
    text = "\n\n".join(paragraphs).strip()
    notes = _validate_chunk_text(text, config)
    status = "ok" if not notes else "review"

    return Chunk(
        id=f"chunk-{len(chunks) + 1:04d}",
        text=text,
        source=source,
        start_paragraph=start_paragraph,
        end_paragraph=end_paragraph,
        word_count=_count_words(text),
        validation_status=status,
        validation_notes=notes,
    )


def _merge_last_chunk(last_chunk: Chunk, extra_paragraphs: list[str], config: ChunkingConfig) -> Chunk:
    text = "\n\n".join([last_chunk.text, *extra_paragraphs]).strip()
    notes = _validate_chunk_text(text, config)

    return Chunk(
        id=last_chunk.id,
        text=text,
        source=last_chunk.source,
        start_paragraph=last_chunk.start_paragraph,
        end_paragraph=last_chunk.end_paragraph + len(extra_paragraphs),
        word_count=_count_words(text),
        validation_status="ok" if not notes else "review",
        validation_notes=notes,
    )


def _build_overlap(paragraphs: list[str], overlap_sentences: int) -> list[str]:
    if overlap_sentences <= 0 or not paragraphs:
        return []

    sentences = _split_sentences(paragraphs[-1])
    overlap = " ".join(sentences[-overlap_sentences:]).strip()
    return [overlap] if overlap else []


def _split_long_paragraph(paragraph: str, max_words: int) -> list[str]:
    sentence_chunks = []
    current_sentences = []

    for sentence in _split_sentences(paragraph):
        candidate = " ".join([*current_sentences, sentence])
        if current_sentences and _count_words(candidate) > max_words:
            sentence_chunks.append(" ".join(current_sentences).strip())
            current_sentences = [sentence]
        else:
            current_sentences.append(sentence)

    if current_sentences:
        sentence_chunks.append(" ".join(current_sentences).strip())

    return sentence_chunks


def _validate_chunk_text(text: str, config: ChunkingConfig) -> list[str]:
    notes = []
    word_count = _count_words(text)

    if word_count > config.max_words:
        notes.append(f"Chunk exceeds max_words ({word_count}>{config.max_words}).")

    if word_count < config.min_words:
        notes.append(f"Chunk is below min_words ({word_count}<{config.min_words}).")

    if re.search(r"[,;:]$", text):
        notes.append("Chunk ends with punctuation that suggests an incomplete idea.")

    if re.search(r"\b(y|o|de|del|la|el|los|las|que|para|con|por)$", text, flags=re.IGNORECASE):
        notes.append("Chunk ends with a connector or article.")

    return notes


def _split_sentences(text: str) -> list[str]:
    # Simple Spanish-friendly sentence split. It keeps punctuation at sentence end.
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def _count_words(text: str) -> int:
    return len(re.findall(r"\b[\wáéíóúÁÉÍÓÚñÑüÜ]+\b", text))
