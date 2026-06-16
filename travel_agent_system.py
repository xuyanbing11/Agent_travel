# travel_agent_pure_api.py - 纯API调用版本（无模拟数据）
import os
import json
import re
import openai
from typing import TypedDict, List, Dict, Any
from langchain_openai import ChatOpenAI
from typing import Optional
from langgraph.graph import StateGraph, END
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool

# ==================== 配置 ====================
os.environ["OPENAI_API_KEY"] = "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"  # 必须配置
os.environ["SERPER_API_KEY"] = "eb01556659c5d0c5148340b24194609361ead554"  # 必须配置
os.environ["DEEPSEEK_API_KEY"] = "sk-642bb22e6a544877b12f0e7df8eef7c7"  # 必须配置

# 检查所有必需的 API Keys
required_keys = ["OPENAI_API_KEY", "SERPER_API_KEY", "DEEPSEEK_API_KEY"]
missing_keys = [key for key in required_keys if not os.environ.get(key)]

if missing_keys:
    raise ValueError(f"❌ 缺少必需的API Key：{missing_keys}。请先配置所有API Key再运行。")


class TravelState(TypedDict):
    user_request: str
    parsed_info: Dict[str, Any]
    daily_plan: List[Dict[str, Any]]
    budget: Dict[str, Any]
    messages: List[str]
    final_report: str
    search_context: str


# ==================== 三大模型初始化 ====================
def init_llm_parser() -> ChatOpenAI:
    """智能体1专用：解析用户意图 - 使用 GPT-3.5"""
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=500,
        api_key=os.environ["OPENAI_API_KEY"]
    )


def init_llm_planner() -> ChatOpenAI:
    """智能体2专用：深度规划行程 - 使用 DeepSeek"""
    return ChatOpenAI(
        model="deepseek-chat",
        temperature=0.3,
        max_tokens=2000,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com/v1"
    )


def init_llm_reporter() -> ChatOpenAI:
    """智能体4专用：报告润色 - 使用 DeepSeek（不同参数）"""
    return ChatOpenAI(
        model="deepseek-chat",
        temperature=0.7,
        max_tokens=1500,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com/v1"
    )


# 全局模型实例
LLM_PARSER = init_llm_parser()
LLM_PLANNER = init_llm_planner()
LLM_REPORTER = init_llm_reporter()


# ==================== 搜索工具初始化 ====================
def init_search_tool():
    """初始化网络搜索工具"""
    try:
        search = GoogleSerperAPIWrapper(serper_api_key=os.environ["SERPER_API_KEY"])
        search_tool = Tool(
            name="WebSearch",
            description="搜索最新的景点信息、开放时间、用户评价、实时攻略。",
            func=search.run
        )
        print("✅ Serper 搜索工具初始化成功")
        return search_tool
    except Exception as e:
        raise RuntimeError(f"搜索工具初始化失败: {e}")


SEARCH_TOOL = init_search_tool()


# ==================== 真实工具调用 ====================
def real_gpt_parser(state: TravelState):
    """使用真实 GPT 解析用户意图"""
    print("🧠 智能体1 (GPT-3.5) 正在解析用户意图...")

    prompt = f"""
    作为旅行规划助手，请仔细分析用户的旅行请求并提取关键信息。

    用户请求：{state['user_request']}

    请特别注意：用户可能没有明确说天数，请根据上下文推断合理的旅行天数。
    例如："我想去北京玩"可以推断为3-5天，"周末去上海"可以推断为2天。

    请提取以下信息，以JSON格式返回：
    {{
        "destination": "城市名",
        "days": 天数（必须是正整数，如果无法推断则默认为3）,
        "people": 人数（必须是正整数，如果无法推断则默认为1）,
        "budget_level": "经济/中等/豪华"（根据预算金额或描述判断）,
        "travel_style": "家庭游/情侣游/朋友游/独自旅行",
        "interests": ["兴趣1", "兴趣2"]（如果没有明确兴趣则返回空列表）,
        "special_requirements": ["特殊要求1", "特殊要求2"]
    }}

    重要规则：
    1. days 字段必须是正整数，最小为1，最大为30
    2. people 字段必须是正整数，最小为1
    3. 如果用户没有明确提及天数，请根据城市和预算推断一个合理的默认值（如3-5天）

    注意：确保JSON格式正确，可以直接用json.loads()解析。
    """

    print("🔄 正在调用 GPT-3.5 解析用户请求...")
    response = LLM_PARSER.invoke(prompt)

    # 清理响应，提取 JSON
    content = response.content.strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    parsed_info = json.loads(content)

    # 数据验证和修正逻辑
    print(f"🛠️  开始验证解析结果...")

    # 1. 确保天数是正整数
    if parsed_info.get('days', 0) <= 0:
        print(f"⚠️  解析的天数无效 ({parsed_info.get('days')})，设置为默认值3")
        parsed_info['days'] = 3

    # 2. 确保人数是正整数
    if parsed_info.get('people', 0) <= 0:
        print(f"⚠️  解析的人数无效 ({parsed_info.get('people')})，设置为默认值1")
        parsed_info['people'] = 1

    # 3. 确保预算等级合理
    valid_budget_levels = ["经济", "中等", "豪华"]
    if parsed_info.get('budget_level') not in valid_budget_levels:
        budget_text = state['user_request']
        if any(word in budget_text for word in ["豪华", "奢侈", "高端"]):
            parsed_info['budget_level'] = "豪华"
        elif any(word in budget_text for word in ["中等", "适中", "一般"]):
            parsed_info['budget_level'] = "中等"
        else:
            parsed_info['budget_level'] = "经济"

    # 4. 确保目的地不为空
    if not parsed_info.get('destination') or parsed_info['destination'] == "":
        # 尝试从请求中提取城市
        cities = ["北京", "上海", "广州", "深圳", "杭州", "西安", "成都", "重庆", "南京", "武汉"]
        for city in cities:
            if city in state['user_request']:
                parsed_info['destination'] = city
                break
        if not parsed_info.get('destination'):
            parsed_info['destination'] = "北京"

    print(f"✅ 验证后的解析结果：{parsed_info}")
    message = f"📍 智能解析完成：{parsed_info['destination']} {parsed_info['days']}日游，{parsed_info['people']}人"

    return {
        "parsed_info": parsed_info,
        "messages": state["messages"] + [message],
        "search_context": ""
    }


# ==================== 增强的智能体2：集成实时搜索 ====================
def search_attractions_online(destination: str, interests: List[str], days: int):
    """使用Serper API在线搜索景点信息"""
    # 验证天数
    valid_days = max(1, min(days, 30))
    if days != valid_days:
        print(f"⚠️  调整搜索天数：从{days}天调整为{valid_days}天")
        days = valid_days

    try:
        print(f"\n🔍 正在为【{destination}】搜索实时旅行信息...")
        search_results = []

        # 1. 搜索热门景点
        hot_query = f"{destination} 必去景点 top10 2024最新"
        print(f"   搜索热门: {hot_query}")
        hot_result = SEARCH_TOOL.run(hot_query)
        if hot_result:
            search_results.append(f"【热门景点】{hot_result[:300]}")

        # 2. 根据兴趣搜索
        for interest in interests[:2]:
            interest_query = f"{destination} {interest} 景点 最新攻略"
            print(f"   搜索兴趣点『{interest}』: {interest_query}")
            interest_result = SEARCH_TOOL.run(interest_query)
            if interest_result:
                search_results.append(f"【{interest}】{interest_result[:250]}")

        # 3. 搜索行程建议
        itinerary_query = f"{destination} {days}天行程安排 最新"
        print(f"   搜索行程建议: {itinerary_query}")
        itinerary_result = SEARCH_TOOL.run(itinerary_query)
        if itinerary_result:
            search_results.append(f"【行程参考】{itinerary_result[:200]}")

        if search_results:
            full_context = "\n".join(search_results)
            print(f"✅ 成功获取{len(search_results)}条网络信息")
            return True, full_context[:800]
        else:
            raise ValueError("未搜索到有效信息")

    except Exception as e:
        raise RuntimeError(f"网络搜索失败: {e}")


def planning_agent_with_search(state: TravelState):
    """增强版行程规划智能体 - 集成实时搜索"""
    info = state["parsed_info"]
    destination = info["destination"]
    days = info["days"]
    interests = info.get("interests", ["观光"])

    print(f"\n🎯 智能体2（增强版）启动")
    print(f"   目的地: {destination}, 天数: {days}, 兴趣: {interests}")

    # 网络搜索
    search_success, search_context = search_attractions_online(destination, interests, days)

    # 基于搜索结果生成行程
    print("🧠 智能体2 (DeepSeek) 正在从搜索结果中提取并规划具体景点...")
    itinerary = generate_itinerary_with_search(destination, days, interests, search_context)

    message = f"📅 智能规划：基于实时网络信息，为{destination}制定了{days}天行程"

    return {
        "daily_plan": itinerary,
        "messages": state["messages"] + [message],
        "search_context": search_context
    }


def generate_itinerary_with_search(destination: str, days: int, interests: List[str], search_context: str):
    """基于网络搜索结果生成智能行程"""
    planning_prompt = f"""
    你是一个专业的旅行规划师。请基于以下关于{destination}的网络搜索结果，为一位游客规划一个精确到每日上午、下午具体景点的{days}天行程。
    用户明确表示的兴趣是：{interests}。

    【网络搜索结果原文】
    {search_context[:3000]}

    【你的任务】
    1. 从上述搜索结果中，提取出至少 {days * 2} 个具体的、值得游览的景点或活动名称。
    2. 将这些景点合理地分配到 {days} 天的行程中，每天安排上午和下午各一个主要景点或活动。
    3. 考虑景点的热门程度、地理位置（尽量将靠近的排在一天）、开放时间以及用户的兴趣。
    4. 如果搜索结果中提到的景点数量不足，可以基于你对{destination}的了解，补充最著名的景点。

    【输出格式要求】
    请严格按照以下JSON数组格式输出，不要有任何其他解释：
    [
      {{
        "day": 1,
        "morning": "上午要游览的具体景点或活动名称",
        "afternoon": "下午要游览的具体景点或活动名称",
        "evening": "当地特色晚餐/自由活动",
        "desc": "简短的一句话描述，突出今天行程的亮点",
        "data_source": "网络实时信息"
      }},
      ...  // 第2天，第3天...
    ]
    请确保 `morning` 和 `afternoon` 字段的值是**具体的景点名或活动名**。
    """

    try:
        response = LLM_PLANNER.invoke(planning_prompt)
        content = response.content.strip()

        # 清理响应，提取JSON数组
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        itinerary = json.loads(content)
        print(f"✅ DeepSeek已生成精确到景点的{len(itinerary)}天行程。")
        return itinerary

    except json.JSONDecodeError as e:
        raise RuntimeError(f"DeepSeek返回的行程JSON格式解析失败: {e}")
    except Exception as e:
        raise RuntimeError(f"调用DeepSeek规划行程失败: {e}")


# ==================== 预算智能体 ====================
def budget_agent(state: TravelState):
    """预算估算智能体"""
    info = state["parsed_info"]

    base_cost = {
        "经济": {"hotel": 200, "food": 80, "ticket": 60, "transport": 30},
        "中等": {"hotel": 400, "food": 150, "ticket": 100, "transport": 50},
        "豪华": {"hotel": 800, "food": 300, "ticket": 200, "transport": 100}
    }

    level = info.get("budget_level", "中等")
    rates = base_cost.get(level, base_cost["中等"])

    days = info["days"]
    people = info["people"]

    calculations = {
        "住宿": rates["hotel"] * days,
        "餐饮": rates["food"] * days * people,
        "门票": rates["ticket"] * days * people,
        "交通": rates["transport"] * days * people,
        "购物": 300 * people,
        "应急": 200 * people
    }

    total = sum(calculations.values())
    per_person = total / people if people > 0 else total

    budget = {
        "明细": calculations,
        "总计": round(total, 2),
        "人均": round(per_person, 2),
        "货币": "人民币",
        "预算等级": level
    }

    message = f"💰 预算估算：总计{budget['总计']}元，人均{budget['人均']}元"

    return {
        "budget": budget,
        "messages": state["messages"] + [message]
    }


# ==================== 报告智能体 ====================
def report_agent(state: TravelState):
    """报告生成智能体 - 使用 DeepSeek 进行报告优化"""
    info = state["parsed_info"]
    plan = state["daily_plan"]
    budget = state["budget"]
    search_context = state.get("search_context", "")

    print("🖋️  智能体4 (DeepSeek-润色版) 正在优化报告...")

    # 构建原始报告数据
    report_data = {
        "destination": info['destination'],
        "days": info['days'],
        "people": info['people'],
        "travel_style": info.get('travel_style', '常规'),
        "interests": info.get('interests', ['常规旅游']),
        "itinerary": plan,
        "budget": budget,
        "search_context_preview": search_context[:200] if search_context else "",
        "special_requirements": info.get("special_requirements", [])
    }

    polish_prompt = f"""
    你是一位专业的旅行文案编辑，擅长将技术性报告转化为生动有趣的旅行指南。

    【编辑任务】
    请基于以下旅行数据，创建一个生动、专业、详细的旅行规划报告：

    【旅行数据】
    目的地：{report_data['destination']}
    天数：{report_data['days']}天
    人数：{report_data['people']}人
    旅行风格：{report_data['travel_style']}
    兴趣点：{', '.join(report_data['interests'])}
    特殊要求：{', '.join(report_data['special_requirements']) if report_data['special_requirements'] else '无'}

    【每日行程安排】
    {json.dumps(report_data['itinerary'], ensure_ascii=False, indent=2)}

    【预算明细】
    预算等级：{report_data['budget']['预算等级']}
    {chr(10).join([f'    {item}: {amount}元' for item, amount in report_data['budget']['明细'].items()])}
    总计：{report_data['budget']['总计']}元
    人均：{report_data['budget']['人均']}元

    【网络信息参考】
    {report_data['search_context_preview']}

    【报告要求】
    1. 保持所有原始数据绝对不变（景点名称、预算数字、天数、人数等）
    2. 优化语言表达，使描述更生动、更具吸引力
    3. 结构清晰，包含：欢迎语、基本信息、详细行程、预算明细、贴心建议等部分
    4. 可以添加适当的表情符号让报告更活泼
    5. 在行程描述中突出每个景点的特色
    6. 添加实用的旅行建议和注意事项

    【输出格式】
    直接输出优化后的完整报告，使用中文，不要有任何额外解释。
    """

    try:
        polished_response = LLM_REPORTER.invoke(polish_prompt)
        final_report = polished_response.content
        polish_msg = "📝 报告已使用DeepSeek优化完成"

    except Exception as e:
        raise RuntimeError(f"报告润色失败: {e}")

    return {
        "final_report": final_report,
        "messages": state["messages"] + [polish_msg]
    }


# ==================== 构建智能体系统 ====================
def build_pure_api_system():
    """构建纯API调用系统（无模拟数据）"""

    print("=" * 60)
    print("🚀 纯API多智能体旅行规划系统")
    print("=" * 60)

    workflow = StateGraph(TravelState)

    # 添加智能体
    workflow.add_node("智能解析", real_gpt_parser)
    workflow.add_node("规划行程", planning_agent_with_search)
    workflow.add_node("估算预算", budget_agent)
    workflow.add_node("生成报告", report_agent)

    # 设置流程
    workflow.set_entry_point("智能解析")
    workflow.add_edge("智能解析", "规划行程")
    workflow.add_edge("规划行程", "估算预算")
    workflow.add_edge("估算预算", "生成报告")
    workflow.add_edge("生成报告", END)

    system = workflow.compile()
    print("✅ 纯API系统构建完成！")
    return system


# ==================== 主程序 ====================
def main():
    """运行纯API系统"""
    print("=" * 60)
    print("🔧 三模型智能旅行规划系统（纯API版本）")
    print("=" * 60)
    print("模型架构：")
    print("  • 智能体1 (解析): GPT-3.5-Turbo")
    print("  • 智能体2 (规划): DeepSeek-V3")
    print("  • 智能体3 (预算): 规则引擎")
    print("  • 智能体4 (报告): DeepSeek-润色版")
    print("  • 实时搜索: Serper API")
    print("=" * 60)

    # 用户输入
    print("\n💬 请输入旅行需求（示例：我想去上海玩3天，2个人，预算中等，喜欢迪士尼）")
    user_input = input("> ").strip()

    if not user_input:
        print("请输入有效的旅行需求！")
        return

    # 构建并运行系统
    try:
        system = build_pure_api_system()

        initial_state = {
            "user_request": user_input,
            "parsed_info": {},
            "daily_plan": [],
            "budget": {},
            "messages": ["系统启动"],
            "final_report": "",
            "search_context": ""
        }

        print("\n" + "=" * 40)
        print("🔄 开始智能体协作流程...")
        print("=" * 40)

        result = system.invoke(initial_state)

        # 显示处理日志
        print("\n📊 智能体处理日志：")
        for i, msg in enumerate(result["messages"], 1):
            print(f"  {i}. {msg}")

        # 显示最终结果
        print("\n" + "=" * 60)
        print("🎉 规划完成！")
        print("=" * 60)
        print(result["final_report"])

        # 保存结果
        filename = f"travel_plan_pure_api.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(result["final_report"])
        print(f"\n💾 报告已保存：{filename}")

        # 保存搜索上下文
        if result.get("search_context"):
            with open("search_context_pure_api.txt", "w", encoding="utf-8") as f:
                f.write(result["search_context"])
            print("💾 网络搜索原始数据已保存：search_context_pure_api.txt")

    except Exception as e:
        print(f"\n❌ 系统运行失败: {e}")
        print("请检查：")
        print("  1. 所有API Key是否正确配置")
        print("  2. 网络连接是否正常")
        print("  3. API配额是否充足")


if __name__ == "__main__":
    main()