"""Data models for the chunking pipeline."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ChunkRecord:
    """Final output record for a single chunk, ready for storage."""

    chunk_id: str  # e.g. "doc_id_chunk_0"
    doc_id: str  # Associated document ID
    content: str  # Final chunk text
    code_blocks: str  # Extracted pure code text (for BM25 code search)
    content_vector: List[float] = field(default_factory=list)  # Embedding vector
    title: str = ""  # Original document title
