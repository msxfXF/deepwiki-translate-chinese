import asyncio
from pathlib import Path

import pytest

from translator.chunker import MarkdownChunk, chunk_markdown
from translator.main import translate_file


class FakeTranslationClient:
    """A stub translation client that annotates translated text."""

    def translate_text(self, text: str, temperature: float = 0.2) -> str:  # noqa: ARG002
        return f"[ZH]{text}[/ZH]"


def test_chunk_markdown_preserves_code_blocks() -> None:
    content = """# Title

Some text before code.

```
print('hello')
```

More text after code.
"""

    chunks = chunk_markdown(content, max_chars=50)

    assert chunks == [
        MarkdownChunk("text", "# Title\n\nSome text before code.\n\n"),
        MarkdownChunk("code", "```\nprint('hello')\n```\n"),
        MarkdownChunk("text", "\nMore text after code.\n"),
    ]


def test_chunk_markdown_splits_long_text() -> None:
    content = "A" * 10 + "B" * 15
    chunks = chunk_markdown(content, max_chars=12)

    assert [c.kind for c in chunks] == ["text", "text", "text"]
    assert [c.content for c in chunks] == ["A" * 10 + "B" * 2, "B" * 12, "B"]


@pytest.mark.asyncio
async def test_translate_file_writes_translated_markdown(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    markdown = """Intro line.

Code stays:
```
echo test
```

Visit [Example](https://example.com).
"""
    source_file = input_dir / "sample.md"
    source_file.write_text(markdown, encoding="utf-8")

    client = FakeTranslationClient()
    semaphore = asyncio.Semaphore(2)

    await translate_file(
        path=source_file,
        input_root=input_dir,
        output_root=output_dir,
        client=client,
        max_chars=500,
        semaphore=semaphore,
    )

    translated = (output_dir / "sample.md").read_text(encoding="utf-8")

    expected = """[ZH]Intro line.\n\nCode stays:\n[/ZH]```
echo test
```
[ZH]
Visit [Example](https://example.com).\n[/ZH]"""

    assert translated == expected

