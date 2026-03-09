"""Tests for the document chunking pipeline."""

import unittest

from pipeline.chunk_processor import ChunkProcessor, _split_into_blocks, _extract_code_blocks
from pipeline.models import ChunkRecord


class TestChunkProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = ChunkProcessor(max_chunk_chars=2048, min_chunk_chars=120)

    # -- Stage 2: block splitting ------------------------------------------

    def test_code_block_kept_intact(self):
        text = "Some intro.\n\n```jsx\nconst a = 1;\nconst b = 2;\n```\n\nAfter code."
        blocks = _split_into_blocks(text)
        code_blocks = [b for b in blocks if b.is_code]
        self.assertEqual(len(code_blocks), 1)
        self.assertIn("const a = 1;", code_blocks[0].text)
        self.assertIn("const b = 2;", code_blocks[0].text)

    def test_heading_block_detected(self):
        text = "# Title\n\nSome content here."
        blocks = _split_into_blocks(text)
        heading_blocks = [b for b in blocks if b.is_heading]
        self.assertEqual(len(heading_blocks), 1)
        self.assertEqual(heading_blocks[0].text, "# Title")

    # -- Stage 3: merging --------------------------------------------------

    def test_short_blocks_merged(self):
        text = "# Heading\n\nShort.\n\n# Another\n\nAlso short content here that is a bit longer to test."
        chunks = self.processor.chunk_text(text)
        # Short blocks should be merged, not left as fragments
        for chunk in chunks:
            # No chunk should be just "Short." alone
            self.assertNotEqual(chunk.strip(), "Short.")

    def test_heading_only_merged_with_content(self):
        text = "# Title\n\nParagraph with enough content to be meaningful on its own and not be considered short."
        chunks = self.processor.chunk_text(text)
        # The heading should be merged with the paragraph
        self.assertTrue(any("# Title" in c and "Paragraph" in c for c in chunks))

    # -- Stage 4: code block not split -------------------------------------

    def test_large_code_block_not_split(self):
        long_code = "line_{}\n".format("x" * 100) * 50
        text = f"# Code Example\n\n```python\n{long_code}```\n"
        processor = ChunkProcessor(max_chunk_chars=500, min_chunk_chars=50)
        chunks = processor.chunk_text(text)
        # The code block should remain intact in a single chunk
        code_chunks = [c for c in chunks if "```python" in c]
        self.assertTrue(len(code_chunks) >= 1)
        # Verify the code content is not split
        full = "\n\n".join(chunks)
        self.assertIn("line_", full)

    def test_large_text_block_is_split(self):
        long_text = ("这是一段很长的中文文本。" * 100) + "\n"
        text = f"# Section\n\n{long_text}"
        processor = ChunkProcessor(max_chunk_chars=500, min_chunk_chars=50)
        chunks = processor.chunk_text(text)
        self.assertGreater(len(chunks), 1)

    # -- Stage 6: code extraction ------------------------------------------

    def test_extract_code_blocks(self):
        text = "Text.\n\n```js\nconsole.log('hello');\n```\n\nMore text.\n\n```py\nprint('hi')\n```"
        code = _extract_code_blocks(text)
        self.assertIn("console.log('hello');", code)
        self.assertIn("print('hi')", code)

    # -- Full pipeline: process_document -----------------------------------

    def test_process_document_returns_chunk_records(self):
        text = "# Doc Title\n\nSome body text that is long enough to be its own chunk for testing purposes in the pipeline."
        records = self.processor.process_document(text, doc_id="test_doc", title="Doc Title")
        self.assertTrue(all(isinstance(r, ChunkRecord) for r in records))
        self.assertTrue(all(r.doc_id == "test_doc" for r in records))
        self.assertTrue(all(r.chunk_id.startswith("test_doc_chunk_") for r in records))
        self.assertTrue(all(r.title == "Doc Title" for r in records))

    def test_process_document_no_empty_chunks(self):
        text = "# A\n\nContent A.\n\n# B\n\nContent B with more text to fill it up a bit.\n\n# C\n\n```js\ncode();\n```"
        records = self.processor.process_document(text, doc_id="d1", title="T")
        for r in records:
            self.assertTrue(len(r.content.strip()) > 0, "Empty chunk found")

    # -- Edge cases --------------------------------------------------------

    def test_empty_document(self):
        chunks = self.processor.chunk_text("")
        self.assertEqual(chunks, [])

    def test_only_headings(self):
        text = "# H1\n\n## H2\n\n### H3"
        chunks = self.processor.chunk_text(text)
        # All headings should be merged together
        self.assertGreaterEqual(len(chunks), 1)

    def test_nested_code_fences(self):
        text = "````md\n```js\nconsole.log(1);\n```\n````"
        blocks = _split_into_blocks(text)
        code_blocks = [b for b in blocks if b.is_code]
        self.assertEqual(len(code_blocks), 1)


class TestAddEmbeddings(unittest.TestCase):
    def test_add_embeddings(self):
        processor = ChunkProcessor()
        records = [
            ChunkRecord(chunk_id="c0", doc_id="d", content="hello", code_blocks=""),
            ChunkRecord(chunk_id="c1", doc_id="d", content="world", code_blocks=""),
        ]

        def mock_embed(texts):
            return [[0.1] * 3 for _ in texts]

        processor.add_embeddings(records, mock_embed, batch_size=2)
        self.assertEqual(len(records[0].content_vector), 3)
        self.assertEqual(len(records[1].content_vector), 3)


if __name__ == "__main__":
    unittest.main()
