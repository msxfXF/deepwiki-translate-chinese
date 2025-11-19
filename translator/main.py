from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import List

from .chunker import MarkdownChunk, chunk_markdown
from .translator import TranslationClient


async def translate_chunk(client: TranslationClient, chunk: MarkdownChunk, semaphore: asyncio.Semaphore) -> str:
    if chunk.kind != "text":
        return chunk.content
    async with semaphore:
        return await asyncio.to_thread(client.translate_text, chunk.content)


def rebuild_content(chunks: List[MarkdownChunk], translations: List[str]) -> str:
    pieces: List[str] = []
    translation_idx = 0
    for chunk in chunks:
        if chunk.kind == "text":
            pieces.append(translations[translation_idx])
            translation_idx += 1
        else:
            pieces.append(chunk.content)
    return "".join(pieces)


async def translate_file(
    path: Path,
    input_root: Path,
    output_root: Path,
    client: TranslationClient,
    max_chars: int,
    semaphore: asyncio.Semaphore,
) -> None:
    content = path.read_text(encoding="utf-8")
    chunks = chunk_markdown(content, max_chars=max_chars)
    text_chunks = [chunk for chunk in chunks if chunk.kind == "text"]

    tasks = [translate_chunk(client, chunk, semaphore) for chunk in text_chunks]
    translations = await asyncio.gather(*tasks)
    rebuilt = rebuild_content(chunks, translations)

    relative_path = path.relative_to(input_root)
    target_path = output_root / relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(rebuilt, encoding="utf-8")
    print(f"Translated {path} -> {target_path}")


def collect_markdown_files(root: Path, exclude: Path | None = None) -> List[Path]:
    files: List[Path] = []
    for path in root.rglob("*.md"):
        if not path.is_file():
            continue
        if exclude and path.is_relative_to(exclude):
            continue
        files.append(path)
    return files


async def main_async(args: argparse.Namespace) -> None:
    client = TranslationClient(api_key=args.api_key, model=args.model)
    semaphore = asyncio.Semaphore(args.concurrency)

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    files = collect_markdown_files(input_root, exclude=output_root)

    if not files:
        print("No Markdown files found; nothing to translate.")
        return

    tasks = [
        translate_file(path, input_root, output_root, client, args.max_chars, semaphore)
        for path in files
    ]
    await asyncio.gather(*tasks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate Markdown files to Chinese using GLM-4.5-Flash")
    parser.add_argument("--input", required=True, help="Input directory containing Markdown files")
    parser.add_argument("--output", required=True, help="Output directory for translated files")
    parser.add_argument("--max-chars", type=int, default=1200, help="Maximum characters per translation chunk")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent translation requests")
    parser.add_argument("--model", default="glm-4.5-flash", help="Model name to use")
    parser.add_argument("--api-key", dest="api_key", default=None, help="Override API key (otherwise uses ZHIPU_API_KEY env var)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
