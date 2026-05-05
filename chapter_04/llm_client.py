import os
from typing import Dict, List

from openai import OpenAI

MODEL = "kimi-k2.5"
DEFAULT_BASE = "https://api.moonshot.cn/v1"


class HelloAgentsLLM:
    """固定 kimi-k2.5；密钥与地址仅从环境变量（或构造参数）读取。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        key = (api_key or os.getenv("MOONSHOT_API_KEY", "")).strip()
        if not key:
            raise ValueError("请执行: export MOONSHOT_API_KEY='sk-...'")
        base = (base_url or os.getenv("MOONSHOT_BASE_URL") or DEFAULT_BASE).strip().rstrip("/")
        self.model = MODEL
        self.client = OpenAI(api_key=key, base_url=base, timeout=timeout)

    def think(self, messages: List[Dict[str, str]]) -> str | None:
        # kimi-k2.5 当前接口要求 temperature 只能为 1
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=1.0,
                stream=True,
            )

            # 处理流式响应
            buf: list[str] = []
            for chunk in stream:
                if not chunk.choices:
                    continue
                piece = chunk.choices[0].delta.content or ""
                print(piece, end="", flush=True)
                buf.append(piece)
            print()
            return "".join(buf)
        except Exception as e:
            print(f"\n{e}")
            return None


if __name__ == "__main__":
    llm = HelloAgentsLLM()
    msgs = [
        {"role": "system", "content": "You are a helpful assistant that writes Python code."},
        {"role": "user", "content": "写一个快速排序算法"},
    ]
    llm.think(msgs)