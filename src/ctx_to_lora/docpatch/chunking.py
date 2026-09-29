from dataclasses import dataclass
import re


@dataclass(frozen=True)
class TextChunk:
    chunk_id: int
    text: str
    start_token: int
    end_token: int


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_document(
    doc: str,
    chunk_size: int = 512,
    overlap: int = 128,
) -> list[TextChunk]:
    """Split a document into word-token chunks.

    This intentionally uses whitespace tokens so preprocessing can run without
    loading a model tokenizer. Training/precompute code can later retokenize
    each chunk with the base model tokenizer.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    words = normalize_text(doc).split()
    if not words:
        return []

    chunks = []
    step = chunk_size - overlap
    for chunk_id, start in enumerate(range(0, len(words), step)):
        end = min(start + chunk_size, len(words))
        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                text=" ".join(words[start:end]),
                start_token=start,
                end_token=end,
            )
        )
        if end >= len(words):
            break
    return chunks


def chunk_to_docpatch_dict(source_id: str, chunk: TextChunk) -> dict:
    return {
        "source_id": source_id,
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "start_token": chunk.start_token,
        "end_token": chunk.end_token,
    }
