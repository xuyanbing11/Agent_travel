# simple_demo.py - 最简单的 LangGraph 示例
from typing import TypedDict, List
from langgraph.graph import StateGraph, END


# 1. 定义状态（数据在智能体间传递什么）
class MessageState(TypedDict):
    messages: List[str]
    user_input: str
    final_output: str


# 2. 定义第一个智能体
def agent_1(state: MessageState):
    """接收用户输入"""
    user_text = state["user_input"]
    response = f"智能体1收到了：'{user_text}'"
    return {"messages": [response]}


# 3. 定义第二个智能体
def agent_2(state: MessageState):
    """处理并回复"""
    previous = state["messages"][-1]
    response = f"智能体2基于 '{previous}' 进行了处理"
    return {"messages": state["messages"] + [response], "final_output": response}


# 4. 构建智能体图
def build_agent_graph():
    # 创建图
    workflow = StateGraph(MessageState)

    # 添加两个智能体
    workflow.add_node("接收输入", agent_1)
    workflow.add_node("处理回复", agent_2)

    # 设置流程：接收输入 → 处理回复 → 结束
    workflow.set_entry_point("接收输入")
    workflow.add_edge("接收输入", "处理回复")
    workflow.add_edge("处理回复", END)

    # 编译图
    return workflow.compile()


# 5. 运行测试
if __name__ == "__main__":
    print("🚀 开始运行最简单的智能体系统")
    print("-" * 40)

    # 创建智能体系统
    agent_system = build_agent_graph()

    # 准备输入
    test_input = {
        "user_input": "你好，世界！",
        "messages": [],
        "final_output": ""
    }

    # 运行
    result = agent_system.invoke(test_input)

    # 显示结果
    print("📊 处理过程：")
    for i, msg in enumerate(result["messages"], 1):
        print(f"  {i}. {msg}")

    print(f"\n🎯 最终输出：{result['final_output']}")
    print("\n✅ 恭喜！你的第一个智能体系统运行成功！")