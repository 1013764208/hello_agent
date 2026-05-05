"""
仅用 Kimi（OpenAI 兼容接口）单次对话完成「查天气 + 推荐景点」类任务，
与 FirtstAgent.py 里默认的用户任务一致，但不调用任何工具、不跑 Agent 循环。
"""

from __future__ import annotations

import os
import sys

from openai import OpenAI

# 与 FirtstAgent.py 中默认任务一致，便于对比「直接 LLM」与「工具 Agent」
USER_TASK = (
    "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
)

SYSTEM_PROMPT = """
你是一个中文旅行助手。用户会请你根据某地天气推荐游玩方式或景点。

你必须遵守：
1. 你无法访问互联网、天气 API 或任何实时数据，只能根据常识与训练知识作答。
2. 若用户问「今天」「此刻」的天气，请明确写出：你无法提供实时天气，只能给出
   该季节/该地区天气的一般性描述，并建议用户用手机天气 App 或气象局官网确认。
3. 在说明上述限制后，仍请根据「你假设的或一般性的」天气情况，给出 1-2 个
   北京适合该天气的景点建议及简短理由，语气自然友好。
"""


def main():
    api_key = os.getenv("MOONSHOT_API_KEY", os.getenv("LLM_API_KEY", "")).strip()
    base_url = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1").strip().rstrip("/")
    model = os.getenv("LLM_MODEL", "kimi-k2.5").strip()

    if not api_key:
        print(
            "未设置 MOONSHOT_API_KEY 或 LLM_API_KEY，请先 export 后再运行。",
            file=sys.stderr,
        )
        raise SystemExit(1)

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)

    print("用户任务:", USER_TASK)
    print("=" * 60)
    print("正在调用 Kimi（单次对话，无工具）…\n")

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TASK},
        ],
        temperature=1.0,
        stream=False,
    )
    text = resp.choices[0].message.content or ""
    print(text.strip())
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()



"""
用户任务: 你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。
============================================================
正在调用 Kimi（单次对话，无工具）…

你好！很抱歉，我目前**无法获取实时天气数据**，所以查不到北京今天具体的温度、晴雨或空气质量情况。建议你打开手机的天气 App，或者访问中国气象局官网（weather.com.cn）查看准确的今日预报。

不过，如果我**假设今天是个晴朗舒适的日子**（类似北京春秋季节常见的天气），我会推荐你去：

1. **颐和园**  
   这时候昆明湖波光粼粼，在长廊散步或登上万寿山俯瞰都很惬意，既能欣赏皇家园林的精致，又能享受户外的好空气。

2. **故宫**  
   蓝天映衬下的红墙金瓦格外壮丽，适合慢悠悠地逛中轴线，感受历史韵味，拍照光线也会很好。

要是实际查到的天气不佳（比如下雨或雾霾），建议改去**国家博物馆**或**798艺术区**这类室内场所。祝你今天玩得开心！
============================================================
"""


"""
结论：相同的任务，使用 Agent 和直接调用LLM 间，感觉最大的区别是使用工具！
"""
