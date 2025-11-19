from __future__ import annotations

import os
from typing import Optional

from zai import ZhipuAiClient

PROMPT_TEMPLATE = (
    "你是一个专业的 Markdown 翻译助手。"\
    "请将以下 Markdown 文本翻译为简体中文，保持 Markdown 结构和格式不变。"\
    "不要翻译或修改代码块、内联代码、链接文本、URL、文件名、文件路径或图片引用。"\
    "仅翻译自然语言内容，保持列表、标题、表格等格式完整。"\
    "如果遇到不需要翻译的部分，请直接原样保留。"
)


class TranslationClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "glm-4.5-flash"):
        self.api_key = api_key or os.getenv("ZHIPU_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing API key: set ZHIPU_API_KEY environment variable")
        self.model = model
        self.client = ZhipuAiClient(api_key=self.api_key)

    def translate_text(self, text: str, temperature: float = 0.2) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": PROMPT_TEMPLATE},
                {"role": "user", "content": text},
            ],
            thinking={"type": "enabled"},
            stream=False,
            max_tokens=8196,
            temperature=temperature,
        )
        return response.choices[0].message.content
