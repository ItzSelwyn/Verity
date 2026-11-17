"""
chunker.py — Document chunking strategy for Verity Reference Knowledge Base.

Splits long reference passages and source documents into overlapping segments
while preserving sentence and paragraph boundaries to avoid truncating factual claims.

Milestone 1 Specification:
- Target Chunk Size: ~500 tokens
- Target Overlap: ~75 tokens
"""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class TextChunk:
    """Represents an extracted text chunk with positional metadata."""
    text: str
    chunk_index: int
    total_chunks: int
    char_start: int
    char_end: int
    token_count: int


def estimate_tokens(text: str) -> int:
    """
    Fast, reliable heuristic for token count estimation in English text.
    Standard rule: 1 token ≈ 0.75 words (or ~4 characters).
    Splitting on whitespace/punctuation gives a robust approximation.
    """
    if not text:
        return 0
    words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return max(1, len(words))


class DocumentChunker:
    """
    Splits text passages into overlapping chunks of target token size.

    Attributes
    ----------
    target_tokens : int
        Target token size per chunk (default: 500).
    overlap_tokens : int
        Number of overlapping tokens between consecutive chunks (default: 75).
    """

    def __init__(self, target_tokens: int = 500, overlap_tokens: int = 75):
        if overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be strictly smaller than target_tokens.")
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_text(self, text: str) -> List[TextChunk]:
        """
        Split text into overlapping chunks, preferring sentence boundaries.

        Parameters
        ----------
        text : str
            The input passage or document text.

        Returns
        -------
        List[TextChunk]
            List of TextChunk instances with text, index, and offsets.
        """
        if not text or not text.strip():
            return []

        cleaned_text = re.sub(r"\s+", " ", text).strip()
        total_tokens = estimate_tokens(cleaned_text)

        # If the whole passage fits in one chunk, return immediately
        if total_tokens <= self.target_tokens:
            return [
                TextChunk(
                    text=cleaned_text,
                    chunk_index=0,
                    total_chunks=1,
                    char_start=0,
                    char_end=len(cleaned_text),
                    token_count=total_tokens,
                )
            ]

        # Break text into sentences to preserve semantic units
        sentence_pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
        sentences = sentence_pattern.split(cleaned_text)
        if len(sentences) <= 1:
            # Fallback to word splitting if no sentence breaks found
            sentences = cleaned_text.split(". ")
            sentences = [s + "." if not s.endswith(".") else s for s in sentences]

        chunks_data = []
        current_sentences: List[str] = []
        current_token_count = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            s_tokens = estimate_tokens(sentence)

            # If adding this sentence exceeds the target size and we already have content:
            if current_token_count + s_tokens > self.target_tokens and current_sentences:
                chunk_str = " ".join(current_sentences)
                chunks_data.append((chunk_str, current_token_count))

                # Build overlap from the end of current sentences
                overlap_sentences: List[str] = []
                overlap_tokens_accum = 0
                for prev_s in reversed(current_sentences):
                    prev_tokens = estimate_tokens(prev_s)
                    if overlap_tokens_accum + prev_tokens <= self.overlap_tokens or not overlap_sentences:
                        overlap_sentences.insert(0, prev_s)
                        overlap_tokens_accum += prev_tokens
                    else:
                        break

                current_sentences = overlap_sentences + [sentence]
                current_token_count = overlap_tokens_accum + s_tokens
            else:
                current_sentences.append(sentence)
                current_token_count += s_tokens

        if current_sentences:
            chunk_str = " ".join(current_sentences)
            chunks_data.append((chunk_str, current_token_count))

        # Build final TextChunk objects with char offsets
        total_chunks = len(chunks_data)
        result: List[TextChunk] = []
        search_start = 0

        for idx, (chunk_text, token_count) in enumerate(chunks_data):
            char_start = cleaned_text.find(chunk_text[:50], search_start)
            if char_start == -1:
                char_start = search_start
            char_end = char_start + len(chunk_text)
            search_start = max(search_start, char_start + 1)

            result.append(
                TextChunk(
                    text=chunk_text,
                    chunk_index=idx,
                    total_chunks=total_chunks,
                    char_start=char_start,
                    char_end=char_end,
                    token_count=token_count,
                )
            )

        return result
