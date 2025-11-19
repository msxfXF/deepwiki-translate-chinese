from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class MarkdownChunk:
    kind: str
    content: str


CODE_FENCE = "```"


def _flush_buffer(buffer: List[str], chunks: List[MarkdownChunk], max_chars: int) -> None:
    if not buffer:
        return
    current = "".join(buffer)
    start = 0
    while start < len(current):
        end = min(len(current), start + max_chars)
        chunks.append(MarkdownChunk("text", current[start:end]))
        start = end
    buffer.clear()


def chunk_markdown(content: str, max_chars: int = 1200) -> List[MarkdownChunk]:
    """Split markdown content into text chunks and code blocks.

    * Code fences are emitted as-is and never sent to translation.
    * Text chunks are split by character count to keep prompts short.
    """

    chunks: List[MarkdownChunk] = []
    buffer: List[str] = []
    in_code_fence = False
    fence_delimiter = None

    lines = content.splitlines(keepends=True)
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(CODE_FENCE):
            if not in_code_fence:
                _flush_buffer(buffer, chunks, max_chars)
                in_code_fence = True
                fence_delimiter = line.strip()  # retain info for closing
                buffer.append(line)
                continue
            if in_code_fence:
                buffer.append(line)
                chunks.append(MarkdownChunk("code", "".join(buffer)))
                buffer.clear()
                in_code_fence = False
                fence_delimiter = None
                continue

        if in_code_fence:
            buffer.append(line)
            continue

        # outside code fence
        if len("".join(buffer)) + len(line) > max_chars:
            _flush_buffer(buffer, chunks, max_chars)
        buffer.append(line)

    if in_code_fence:
        chunks.append(MarkdownChunk("code", "".join(buffer)))
    else:
        _flush_buffer(buffer, chunks, max_chars)
    return chunks
