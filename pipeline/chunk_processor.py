"""
Document chunking processor for RN Doc Search.

Implements a multi-stage pipeline that splits Markdown documents into
semantically coherent chunks while preserving code block integrity.

Pipeline stages:
  1. Markdown heading-based splitting (MarkdownNodeParser-style)
  2. Fine-grained block splitting (paragraphs, code fences, headings)
  3. Short block merging (eliminate fragment chunks)
  4. Large block splitting (with code-block exemption)
  5. Packing blocks into final chunks
  6. Code feature extraction (for BM25 code index)
  7. Vectorisation (embedding via external API)
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from pipeline.models import ChunkRecord

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_SIZE = 512  # base token-estimate reference
DEFAULT_MAX_CHUNK_CHARS = max(CHUNK_SIZE * 4, 2048)
DEFAULT_MIN_CHUNK_CHARS = 120

# Regex patterns
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_FENCE_OPEN_RE = re.compile(r"^(`{3,}|~{3,})")
_CODE_BLOCK_RE = re.compile(
    r"(`{3,}|~{3,})[^\n]*\n(.*?)(?:\1|$)", re.DOTALL
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；.!?;])\s*")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
@dataclass
class _Block:
    """Intermediate representation of a text block inside the pipeline."""

    text: str
    is_code: bool = False
    is_heading: bool = False


def _is_fence_line(line: str) -> bool:
    """Return True if *line* is a fenced-code delimiter (``` or ~~~)."""
    stripped = line.strip()
    return bool(_FENCE_OPEN_RE.match(stripped))


# ---------------------------------------------------------------------------
# Stage 1 – Markdown heading-based splitting
# ---------------------------------------------------------------------------

def _split_by_headings(text: str) -> List[str]:
    """Split *text* into sections delimited by Markdown headings.

    Each returned string starts with its heading line (except possibly the
    first section which may contain preamble text before any heading).
    """
    sections: List[str] = []
    buf: List[str] = []

    for line in text.splitlines(keepends=True):
        if _HEADING_RE.match(line.rstrip("\n")) and buf:
            sections.append("".join(buf))
            buf = []
        buf.append(line)

    if buf:
        sections.append("".join(buf))

    return sections


# ---------------------------------------------------------------------------
# Stage 2 – Fine-grained block splitting
# ---------------------------------------------------------------------------

def _split_into_blocks(text: str) -> List[_Block]:
    """Split *text* into fine-grained blocks.

    - Fenced code blocks are kept as single intact blocks.
    - Headings become their own blocks.
    - Consecutive non-empty, non-heading lines are grouped as paragraph blocks.
    - Blank lines act as paragraph separators.
    """
    blocks: List[_Block] = []
    lines = text.splitlines(keepends=True)
    i = 0
    buf: List[str] = []

    def _flush_buf():
        content = "".join(buf).strip()
        if content:
            blocks.append(_Block(text=content))
        buf.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip("\n")

        # --- fenced code block ---
        if _FENCE_OPEN_RE.match(stripped.strip()):
            _flush_buf()
            fence_marker = stripped.strip().split()[0]  # e.g. "```"
            code_lines: List[str] = [line]
            i += 1
            while i < len(lines):
                code_lines.append(lines[i])
                if lines[i].strip().startswith(fence_marker) and i > 0:
                    i += 1
                    break
                i += 1
            blocks.append(
                _Block(text="".join(code_lines).strip(), is_code=True)
            )
            continue

        # --- heading ---
        if _HEADING_RE.match(stripped):
            _flush_buf()
            blocks.append(_Block(text=stripped, is_heading=True))
            i += 1
            continue

        # --- blank line => flush ---
        if stripped == "":
            _flush_buf()
            i += 1
            continue

        # --- normal text ---
        buf.append(line)
        i += 1

    _flush_buf()
    return blocks


# ---------------------------------------------------------------------------
# Stage 3 – Short-block & heading merging
# ---------------------------------------------------------------------------

def _merge_short_blocks(
    blocks: List[_Block],
    min_chars: int = DEFAULT_MIN_CHUNK_CHARS,
) -> List[_Block]:
    """Merge heading-only blocks and too-short blocks into neighbours."""
    if not blocks:
        return blocks

    merged: List[_Block] = []

    for blk in blocks:
        # Heading-only blocks always merge forward (appended to next block
        # later), so we tentatively keep them but mark for merge.
        if blk.is_heading:
            merged.append(blk)
            continue

        # Try to merge a preceding heading-only block into this block.
        if merged and merged[-1].is_heading:
            heading = merged.pop()
            blk = _Block(
                text=heading.text + "\n\n" + blk.text,
                is_code=blk.is_code,
            )

        # Merge short non-code blocks into the previous block.
        if (
            len(blk.text) < min_chars
            and not blk.is_code
            and merged
            and not merged[-1].is_code
        ):
            merged[-1] = _Block(
                text=merged[-1].text + "\n\n" + blk.text,
            )
        else:
            merged.append(blk)

    # If the last block is a dangling heading, merge it into the previous.
    if len(merged) >= 2 and merged[-1].is_heading:
        tail = merged.pop()
        merged[-1] = _Block(text=merged[-1].text + "\n\n" + tail.text)

    return merged


# ---------------------------------------------------------------------------
# Stage 4 – Split oversized blocks
# ---------------------------------------------------------------------------

def _split_large_blocks(
    blocks: List[_Block],
    max_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> List[_Block]:
    """Split blocks that exceed *max_chars*.

    Code blocks are exempt – they are never split.
    """
    result: List[_Block] = []

    for blk in blocks:
        if len(blk.text) <= max_chars or blk.is_code:
            result.append(blk)
            continue

        # Try line-level splitting first.
        sub_blocks = _split_text_by_lines(blk.text, max_chars)
        result.extend(sub_blocks)

    return result


def _split_text_by_lines(text: str, max_chars: int) -> List[_Block]:
    """Split *text* at line boundaries; fall back to sentence / hard-wrap."""
    lines = text.split("\n")
    result: List[_Block] = []
    buf: List[str] = []
    buf_len = 0

    def _flush():
        nonlocal buf_len
        content = "\n".join(buf).strip()
        if content:
            result.append(_Block(text=content))
        buf.clear()
        buf_len = 0

    for line in lines:
        line_len = len(line) + 1  # +1 for newline

        if buf and buf_len + line_len > max_chars:
            _flush()

        # Single line still too long – split further.
        if len(line) > max_chars:
            _flush()
            for sub in _split_long_line(line, max_chars):
                result.append(_Block(text=sub))
            continue

        buf.append(line)
        buf_len += line_len

    _flush()
    return result


def _split_long_line(line: str, max_chars: int) -> List[str]:
    """Split a single oversized line by sentence punctuation, then hard-wrap."""
    sentences = _SENTENCE_SPLIT_RE.split(line)
    if len(sentences) > 1:
        parts: List[str] = []
        buf = ""
        for sent in sentences:
            if buf and len(buf) + len(sent) > max_chars:
                parts.append(buf)
                buf = sent
            else:
                buf = buf + sent if buf else sent
        if buf:
            parts.append(buf)
        # If sentence split was effective, return.
        if all(len(p) <= max_chars for p in parts):
            return parts

    # Hard-wrap fallback.
    return [line[i : i + max_chars] for i in range(0, len(line), max_chars)]


# ---------------------------------------------------------------------------
# Stage 5 – Pack blocks into final chunks
# ---------------------------------------------------------------------------

def _pack_chunks(
    blocks: List[_Block],
    max_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    min_chars: int = DEFAULT_MIN_CHUNK_CHARS,
) -> List[str]:
    """Concatenate blocks into chunk strings respecting *max_chars*."""
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def _flush():
        nonlocal buf_len
        text = "\n\n".join(buf).strip()
        if text:
            chunks.append(text)
        buf.clear()
        buf_len = 0

    for blk in blocks:
        blk_len = len(blk.text)
        separator_len = 2 if buf else 0  # "\n\n"

        if buf and buf_len + separator_len + blk_len > max_chars:
            _flush()

        buf.append(blk.text)
        buf_len += blk_len + separator_len

    _flush()

    # Merge trailing fragment into the previous chunk.
    if len(chunks) >= 2 and len(chunks[-1]) < min_chars:
        tail = chunks.pop()
        chunks[-1] = chunks[-1] + "\n\n" + tail

    return chunks


# ---------------------------------------------------------------------------
# Stage 6 – Code-block extraction
# ---------------------------------------------------------------------------

def _extract_code_blocks(text: str) -> str:
    """Return concatenated code content found in fenced blocks of *text*."""
    snippets: List[str] = []
    for m in _CODE_BLOCK_RE.finditer(text):
        code = m.group(2).strip()
        if code:
            snippets.append(code)
    return "\n\n".join(snippets)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ChunkProcessor:
    """Multi-stage Markdown chunking processor.

    Parameters
    ----------
    max_chunk_chars : int
        Soft upper limit on chunk character count (code blocks may exceed it).
    min_chunk_chars : int
        Chunks shorter than this are merged with neighbours.
    """

    def __init__(
        self,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
        min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
    ):
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars

    # -- main entry point ---------------------------------------------------

    def process_document(
        self,
        text: str,
        doc_id: str,
        title: str = "",
    ) -> List[ChunkRecord]:
        """Run the full chunking pipeline on a single document.

        Returns a list of :class:`ChunkRecord` instances (without embeddings –
        call :meth:`add_embeddings` to populate ``content_vector``).
        """
        chunk_texts = self.chunk_text(text)
        records: List[ChunkRecord] = []

        for idx, chunk_content in enumerate(chunk_texts):
            code = _extract_code_blocks(chunk_content)
            records.append(
                ChunkRecord(
                    chunk_id=f"{doc_id}_chunk_{idx}",
                    doc_id=doc_id,
                    content=chunk_content,
                    code_blocks=code,
                    title=title,
                )
            )

        return records

    def chunk_text(self, text: str) -> List[str]:
        """Run stages 1-5 and return the final chunk strings."""
        # Stage 1: heading-based splitting
        sections = _split_by_headings(text)

        # Stage 2: fine-grained block splitting (per section)
        all_blocks: List[_Block] = []
        for section in sections:
            all_blocks.extend(_split_into_blocks(section))

        # Stage 3: merge short / heading-only blocks
        all_blocks = _merge_short_blocks(all_blocks, self.min_chunk_chars)

        # Stage 4: split oversized blocks (code-block exempt)
        all_blocks = _split_large_blocks(all_blocks, self.max_chunk_chars)

        # Stage 5: pack into final chunk strings
        chunks = _pack_chunks(
            all_blocks,
            max_chars=self.max_chunk_chars,
            min_chars=self.min_chunk_chars,
        )

        return chunks

    # -- optional embedding helper -----------------------------------------

    @staticmethod
    def add_embeddings(
        records: List[ChunkRecord],
        embed_fn,
        batch_size: int = 64,
    ) -> None:
        """Populate ``content_vector`` on each record using *embed_fn*.

        Parameters
        ----------
        embed_fn : callable
            ``embed_fn(texts: List[str]) -> List[List[float]]``
        batch_size : int
            Number of texts to send per batch.
        """
        texts = [r.content for r in records]
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors = embed_fn(batch)
            for rec, vec in zip(
                records[start : start + batch_size], vectors
            ):
                rec.content_vector = vec
