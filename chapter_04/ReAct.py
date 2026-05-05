import re
from llm_client import HelloAgentsLLM
from tools import ToolExecutor, search

# (此处省略 REACT_PROMPT_TEMPLATE 的定义)
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下：
{tools}

请严格按照以下格式进行回应：

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一：
- `{{tool_name}}[{{tool_input}}]`：调用一个可用工具。
- `Finish[最终答案]`：当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在`Action:`字段后使用 `Finish[最终答案]` 来输出最终答案。


现在，请开始解决以下问题：
Question: {question}
History: {history}
"""

class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        self.history = []
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(tools=tools_desc, question=question, history=history_str)

            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)
            if not response_text:
                print("错误：LLM未能返回有效响应。"); break

            thought, action = self._parse_output(response_text)
            if thought: print(f"🤔 思考: {thought}")
            if not action: print("警告：未能解析出有效的Action，流程终止。"); break
            
            if action.startswith("Finish"):
                # 如果是 finish 指令，提取最终答案并结束
                final_answer = self._parse_action_input(action)
                print(f"🎉 最终答案: {final_answer}")
                return final_answer
            
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                self.history.append("Observation: 无效的Action格式，请检查。"); continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")
            tool_function = self.tool_executor.getTool(tool_name)
            observation = tool_function(tool_input) if tool_function else f"错误：未找到名为 '{tool_name}' 的工具。"
            
            print(f"👀 观察: {observation}")
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        print("已达到最大步数，流程终止。")
        return None

    def _parse_output(self, text: str):
        # Thought: 匹配到 Action: 或文本末尾
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        # Action: 匹配到文本末尾
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        return (match.group(1), match.group(2)) if match else (None, None)

    def _parse_action_input(self, action_text: str):
        match = re.match(r"\w+\[(.*)\]", action_text, re.DOTALL)
        return match.group(1) if match else ""

if __name__ == '__main__':
    llm = HelloAgentsLLM()
    tool_executor = ToolExecutor()
    search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_desc, search)
    agent = ReActAgent(llm_client=llm, tool_executor=tool_executor)
    question = "2026年华为最新的手机是哪一款？它的主要卖点是什么？"
    agent.run(question)


"""
工具 'Search' 已注册。

--- 第 1 步 ---
Thought: 用户询问的是2026年华为最新的手机。首先需要确认当前年份是2024年，2026年尚未到来，因此2026年的手机应该还未发布。但为了确保信息的准确性，我需要搜索一下是否有关于华为2026年手机的预告、泄露或官方公告，以及目前华为最新的手机产品线信息，以便给用户最准确的回答。

Action: Search[2026年华为手机 华为2026年新机 华为最新手机]
🤔 思考: 用户询问的是2026年华为最新的手机。首先需要确认当前年份是2024年，2026年尚未到来，因此2026年的手机应该还未发布。但为了确保信息的准确性，我需要搜索一下是否有关于华为2026年手机的预告、泄露或官方公告，以及目前华为最新的手机产品线信息，以便给用户最准确的回答。
🎬 行动: Search[2026年华为手机 华为2026年新机 华为最新手机]
🔍 正在执行 [SerpApi] 网页搜索: 2026年华为手机 华为2026年新机 华为最新手机
👀 观察: [1] 2026年华为手机各系列介绍及选购指南（4月份更新） ...
华为今年新机都只有纯血鸿蒙系统版本可选，介意者可以考虑荣耀等其他品牌。 近期更新：已更新4月20日发布的华为Pura90系列。已更新3月23日发布的华为Mate80 Pro Max风驰 ...

[2] 华为手机- 华为官网
探索并选购华为最新手机，了解Mate 系列、Pura 系列、Pocket 系列、nova 系列、畅享系列及相关配件，体验鸿蒙AI 、影像、通信等功能。

[3] 华为春季全场景新品发布会
华为春季全场景新品发布会 ; HUAWEI WATCH GT Runner 2. ￥2588 起 ; HUAWEI Mate 80 Pro Max 风驰版. ￥8499 起 ; 华为畅享90 Pro Max. ￥1699 起 ; 华为畅享90 Plus. ￥1499 起.

[4] 华为Pura 90 系列新机重点参数配置差异公布，一表看懂
IT之家 4 月20 日消息，在今天的华为Pura 系列及全场景新品发布会上，Pura 90 系列正式发布。新机分为三杯：标准版、Pro 和Pro Max，售价4699 元起。

[5] 2026年华为手机各系列介绍及选购指南(3月份更新) ...
其中华为Mate系列属于各方面配置顶配的全水桶旗舰系列。华为Pura系列为主打拍照的旗舰系列，nova系列主打外观及自拍的中端系列，畅享系列主打入门系列。 一 ...

--- 第 2 步 ---
Thought: 从搜索结果来看，2026年华为最新的手机应该是4月20日发布的华为Pura 90系列（比3月发布的Mate 80 Pro Max风驰版更新）。但我需要获取更多关于Pura 90系列主要卖点的详细信息。

Action: Search[华为Pura 90系列 主要卖点 影像 鸿蒙 2026]
🤔 思考: 从搜索结果来看，2026年华为最新的手机应该是4月20日发布的华为Pura 90系列（比3月发布的Mate 80 Pro Max风驰版更新）。但我需要获取更多关于Pura 90系列主要卖点的详细信息。
🎬 行动: Search[华为Pura 90系列 主要卖点 影像 鸿蒙 2026]
🔍 正在执行 [SerpApi] 网页搜索: 华为Pura 90系列 主要卖点 影像 鸿蒙 2026
👀 观察: [1] 华为Pura系列及全场景新品发布会举行，重磅新品悉数亮相
HUAWEI Pura 90系列支持一站式智慧影像，开启智拍新时代，如同随身摄影师与修图师，以实时运动感知采集能力为硬件基础，从按下快门的AI辅助构图、AI姿势推荐、 ...

[2] 4699起交个朋友？华为Pura 90也开始有性价比了
它们这次首发搭载了麒麟9030S 芯片，按照华为官方给到的说法，这颗芯片整机性能相比上一代提升了25%，而且还专门对视频处理能力做了一系列优化。 翻译成人话 ...

[3] 华为Pura 90 系列新机重点参数配置差异公布，一表看懂
为智慧影像而生：华为Pura 90 Pro / Pro Max 首发麒麟9030S 芯片，NPU 图像理解能力提升200% · 发布会场外还有惊喜：鸿蒙HarmonyOS 6.1.0.117 SP6 版本 ...

[4] 鸿蒙最强影像旗舰！华为Pura 90系列发布时间提档：4月见
相比于上一代的Pura 80系列，华为Pura 90系列在屏幕形态上迎来了重大调整。全系机型将回归直屏设计，并提供1.5K的高清分辨率，满足用户对直屏操作体验的追求 ...

[5] 华为Pura 90正式发布！2亿长焦旗舰4699元起
配合HarmonyOS 6.1的AI影像算法，Pura 90系列支持AI辅助构图、AI姿势 ... 对华为粉丝而言，Pura 90系列的最大卖点不只是2亿像素，而是"整套鸿蒙 ...

--- 第 3 步 ---
Thought: 根据搜索结果，2026年华为最新发布的手机是华为Pura 90系列（发布于2026年4月20日），该系列包含标准版、Pro和Pro Max三个版本。相比3月发布的Mate 80 Pro Max风驰版，Pura 90系列发布时间更晚，是目前华为2026年最新的手机产品线。

从搜索结果中可以提取到以下主要卖点：
1. **麒麟9030S芯片**：首发搭载，整机性能相比上一代提升25%，NPU图像理解能力提升200%，专门针对视频处理能力优化
2. **2亿像素长焦影像**：配备2亿像素长焦镜头，配合鸿蒙AI影像算法
3. **鸿蒙HarmonyOS 6.1系统**：搭载最新系统，支持AI辅助构图、AI姿势推荐等一站式智慧影像功能
4. **全系直屏设计**：相比上一代，Pura 90全系回归直屏设计，提供1.5K高清分辨率
5. **智慧影像体验**：支持实时运动感知采集、AI辅助构图、AI姿势推荐等功能，如同随身摄影师与修图师

Action: Finish[2026年华为最新的手机是**华为Pura 90系列**（包含标准版、Pro和Pro Max三款，发布于2026年4月20日）。

**主要卖点包括：**

1. **麒麟9030S芯片**：首发搭载，整机性能较上一代提升25%，NPU图像理解能力提升200%，并针对视频处理能力进行专门优化

2. **2亿像素长焦影像系统**：配备2亿像素长焦镜头，配合HarmonyOS 6.1的AI影像算法，打造鸿蒙最强影像旗舰

3. **一站式智慧影像**：支持AI辅助构图、AI姿势推荐、实时运动感知采集等功能，从按下快门到后期修图提供全流程AI支持

4. **全系直屏设计**：相比上一代Pura 80系列，Pura 90全系回归直屏设计，配备1.5K高清分辨率，满足用户对直屏操作体验的追求

5. **鸿蒙HarmonyOS 6.1系统**：搭载最新鸿蒙系统，起售价4699元，被业内评价为"开始具有性价比"的华为旗舰机型]
🤔 思考: 根据搜索结果，2026年华为最新发布的手机是华为Pura 90系列（发布于2026年4月20日），该系列包含标准版、Pro和Pro Max三个版本。相比3月发布的Mate 80 Pro Max风驰版，Pura 90系列发布时间更晚，是目前华为2026年最新的手机产品线。

从搜索结果中可以提取到以下主要卖点：
1. **麒麟9030S芯片**：首发搭载，整机性能相比上一代提升25%，NPU图像理解能力提升200%，专门针对视频处理能力优化
2. **2亿像素长焦影像**：配备2亿像素长焦镜头，配合鸿蒙AI影像算法
3. **鸿蒙HarmonyOS 6.1系统**：搭载最新系统，支持AI辅助构图、AI姿势推荐等一站式智慧影像功能
4. **全系直屏设计**：相比上一代，Pura 90全系回归直屏设计，提供1.5K高清分辨率
5. **智慧影像体验**：支持实时运动感知采集、AI辅助构图、AI姿势推荐等功能，如同随身摄影师与修图师
🎉 最终答案: 2026年华为最新的手机是**华为Pura 90系列**（包含标准版、Pro和Pro Max三款，发布于2026年4月20日）。

**主要卖点包括：**

1. **麒麟9030S芯片**：首发搭载，整机性能较上一代提升25%，NPU图像理解能力提升200%，并针对视频处理能力进行专门优化

2. **2亿像素长焦影像系统**：配备2亿像素长焦镜头，配合HarmonyOS 6.1的AI影像算法，打造鸿蒙最强影像旗舰

3. **一站式智慧影像**：支持AI辅助构图、AI姿势推荐、实时运动感知采集等功能，从按下快门到后期修图提供全流程AI支持

4. **全系直屏设计**：相比上一代Pura 80系列，Pura 90全系回归直屏设计，配备1.5K高清分辨率，满足用户对直屏操作体验的追求

5. **鸿蒙HarmonyOS 6.1系统**：搭载最新鸿蒙系统，起售价4699元，被业内评价为"开始具有性价比"的华为旗舰机型
"""
