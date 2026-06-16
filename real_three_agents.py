# langgraph_phase3_real_data.py - 三智能体真实数据查询系统
print("=" * 70)
print("三智能体真实数据查询系统")
print("=" * 70)

import os
import sqlite3
import requests
from typing import Literal, TypedDict, Annotated, List
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

# 导入 LangGraph 组件
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


# ===== 第一步：设置真实数据库 (Chinook示例数据库) =====
def setup_real_database():
    """下载并设置一个真实的SQLite数据库到内存中"""
    print("正在设置真实数据库...")
    try:
        # 从GitHub下载Chinook数据库SQL脚本
        url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
        response = requests.get(url)
        sql_script = response.text

        # 创建内存数据库并执行SQL脚本
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        connection.executescript(sql_script)

        # 创建SQLAlchemy引擎
        engine = create_engine(
            "sqlite://",
            creator=lambda: connection,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )

        # 创建LangChain SQLDatabase包装器
        db = SQLDatabase(engine)
        print(f"✅ 数据库已加载，包含表：{db.get_usable_table_names()}")
        return db
    except Exception as e:
        print(f"❌ 数据库设置失败: {e}")
        raise


# 初始化真实数据库
real_db = setup_real_database()


# ===== 第二步：定义真实数据工具 =====
@tool
def query_music_database(query: str) -> str:
    """在真实的音乐数据库上执行SQL查询，返回查询结果"""
    print(f"🎵 [真实数据库工具] 执行查询: {query[:50]}...")
    try:
        # 在真实数据库上执行查询
        result = real_db.run(query)
        print(f"   查询成功，返回 {len(result) if result else 0} 行数据")
        return str(result) if result else "查询未返回结果。"
    except Exception as e:
        error_msg = f"SQL查询错误: {str(e)}"
        print(f"   ❌ {error_msg}")
        return error_msg


@tool
def get_database_schema() -> str:
    """获取当前数据库的架构信息（表和列）"""
    try:
        tables = real_db.get_usable_table_names()
        schema_info = ["数据库架构:"]
        for table in tables:
            create_table_sql = real_db.run(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}';")
            schema_info.append(f"\n{table}: {create_table_sql}")
        return "\n".join(schema_info)
    except Exception as e:
        return f"获取架构失败: {e}"


# ===== 第三步：定义系统状态 =====
class RealDataState(TypedDict):
    """三智能体系统的状态"""
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    task_type: str
    sql_query: str
    query_result: str
    analysis: str


# ===== 第四步：定义三个智能体 =====
# 智能体1：任务分类器
def task_classifier_agent(state: RealDataState) -> dict:
    """分析用户查询，确定任务类型"""
    print(f"\n[智能体1-分类器] 分析查询...")

    user_query = state['user_query']
    messages = state['messages']

    # 使用LLM进行智能分类
    classifier_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    prompt = f"""
    请分析以下用户查询，确定其意图类型：

    用户查询: {user_query}

    可选类型：
    1. data_query - 数据查询（需要从数据库获取信息）
    2. schema_query - 架构查询（需要了解数据库结构）
    3. analysis_request - 分析请求（需要对数据进行解释或总结）
    4. general_question - 一般问题（无需数据库查询）

    只需返回类型名称，不要其他内容。
    """

    response = classifier_model.invoke([HumanMessage(content=prompt)])
    task_type = response.content.strip().lower()

    print(f"   分类结果: {task_type}")

    return {
        "task_type": task_type,
        "messages": messages + [AIMessage(content=f"任务分类为: {task_type}")]
    }


# 智能体2：SQL专家（仅对需要数据库查询的任务工作）
def sql_expert_agent(state: RealDataState) -> dict:
    """根据用户查询生成SQL语句"""
    task_type = state['task_type']

    # 如果不是数据查询任务，跳过SQL生成
    if task_type not in ["data_query", "analysis_request"]:
        print(f"\n[智能体2-SQL专家] 跳过（任务类型: {task_type}）")
        return {"sql_query": "", "messages": state['messages']}

    print(f"\n[智能体2-SQL专家] 生成SQL查询...")

    # 获取数据库架构
    schema = get_database_schema.invoke({})

    # 使用专门优化过的提示词
    sql_model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools([query_music_database])

    prompt = f"""
    你是一个SQL专家。请根据用户查询和数据库架构，生成正确的SQLite查询语句。

    数据库架构:
    {schema}

    用户查询: {state['user_query']}

    重要提示：
    1. Chinook数据库包含音乐商店数据（艺术家、专辑、曲目、顾客、发票等）
    2. 只生成SELECT查询
    3. 确保使用正确的表名和列名
    4. 如果查询涉及中文艺名，注意数据库中使用的是英文名
    5. 如果查询不明确，生成一个合理的查询来获取相关信息

    请只返回SQL语句，不要其他解释。
    """

    response = sql_model.invoke([HumanMessage(content=prompt)])

    # 提取SQL查询
    sql_query = ""
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call['name'] == 'query_music_database':
                sql_query = tool_call['args'].get('query', '')
    elif response.content:
        content = response.content.strip()

        # 移除代码块标记
        if content.startswith('```sql'):
            content = content[6:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]

        # 移除可能的前缀
        prefixes_to_remove = ['sql\n', 'SQL\n', 'sql ', 'SQL ']
        for prefix in prefixes_to_remove:
            if content.startswith(prefix):
                content = content[len(prefix):]

        sql_query = content.strip()

    print(f"   生成的SQL: {sql_query[:80]}..." if len(sql_query) > 80 else f"   生成的SQL: {sql_query}")

    return {
        "sql_query": sql_query,
        "messages": state['messages'] + [response]
    }


# 智能体3：结果分析师
def analysis_agent(state: RealDataState) -> dict:
    """执行查询并分析结果"""
    print(f"\n[智能体3-分析师] 处理任务...")

    task_type = state['task_type']
    sql_query = state['sql_query']
    query_result = state.get('query_result', '')

    # 对于需要数据库查询的任务
    if task_type in ["data_query", "analysis_request"] and sql_query:
        if not query_result:
            print(f"   执行SQL查询...")
            query_result = query_music_database.invoke({"query": sql_query})

            with open("last_query_result.txt", "w", encoding="utf-8") as f:
                f.write(f"SQL: {sql_query}\n\n结果:\n{query_result}")

        # 分析查询结果
        print(f"   分析查询结果...")
        analysis_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

        analysis_prompt = f"""
        用户原始查询: {state['user_query']}

        数据库查询结果:
        {query_result}

        请根据以上结果，生成对用户友好、清晰易懂的回答。
        要点：
        1. 总结关键发现
        2. 如果有数字数据，进行简单分析
        3. 用自然语言解释结果
        4. 如果结果为空或有问题，说明可能的原因
        """

        analysis_response = analysis_model.invoke([HumanMessage(content=analysis_prompt)])
        analysis = analysis_response.content
    else:
        # 对于不需要数据库查询的任务
        print(f"   直接回答一般问题...")
        general_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        response = general_model.invoke(state['messages'])
        analysis = response.content

    print(f"   分析完成")

    return {
        "query_result": query_result,
        "analysis": analysis,
        "messages": [AIMessage(content=analysis)]
    }


# ===== 第五步：定义路由逻辑 =====
def route_after_classifier(state: RealDataState) -> Literal["sql_expert", "analysis_agent", END]:
    """分类器后的路由"""
    task_type = state['task_type']

    if task_type in ["general_question", "schema_query"]:
        print(f"\n[路由] {task_type}任务 -> 直接到分析师")
        return "analysis_agent"
    elif task_type in ["data_query", "analysis_request"]:
        print(f"\n[路由] {task_type}任务 -> 需要SQL专家")
        return "sql_expert"
    else:
        print(f"\n[路由] 未知任务 -> 结束")
        return END


def route_after_sql_expert(state: RealDataState) -> Literal["analysis_agent", END]:
    """SQL专家后的路由"""
    if state['sql_query']:
        print(f"\n[路由] 有SQL查询 -> 到分析师执行和分析")
        return "analysis_agent"
    else:
        print(f"\n[路由] 无SQL查询 -> 结束")
        return END


# ===== 第六步：构建工作流 =====
def main():
    print("\n构建三智能体工作流...")

    workflow = StateGraph(RealDataState)

    # 添加三个智能体节点
    workflow.add_node("classifier", task_classifier_agent)
    workflow.add_node("sql_expert", sql_expert_agent)
    workflow.add_node("analysis_agent", analysis_agent)

    # 设置入口点
    workflow.set_entry_point("classifier")

    # 添加路由边
    workflow.add_conditional_edges(
        "classifier",
        route_after_classifier,
        {
            "sql_expert": "sql_expert",
            "analysis_agent": "analysis_agent",
            "__end__": END
        }
    )

    workflow.add_conditional_edges(
        "sql_expert",
        route_after_sql_expert,
        {
            "analysis_agent": "analysis_agent",
            "__end__": END
        }
    )

    workflow.add_edge("analysis_agent", END)

    # 编译应用
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    print("✅ 三智能体工作流构建完成")

    # ===== 第七步：测试真实数据查询 =====
    print("\n" + "=" * 70)
    print("开始测试真实数据查询系统")
    print("=" * 70)

    test_queries = [
        {
            "query": "数据库里有哪些表？",
            "description": "架构查询（无需SQL）"
        },
        {
            "query": "找出所有来自巴西的顾客",
            "description": "简单数据查询"
        },
        {
            "query": "哪个艺术家发行的专辑最多？",
            "description": "分析型查询"
        },
        {
            "query": "2009年销售额最高的曲目类型是什么？",
            "description": "复杂分析查询"
        },
        {
            "query": "你好，这个系统能做什么？",
            "description": "一般问题"
        }
    ]

    for i, test in enumerate(test_queries, 1):
        print(f"\n{'=' * 50}")
        print(f"测试 {i}: {test['description']}")
        print(f"用户: {test['query']}")
        print('=' * 50)

        try:
            initial_state = {
                "messages": [HumanMessage(content=test['query'])],
                "user_query": test['query'],
                "task_type": "",
                "sql_query": "",
                "query_result": "",
                "analysis": ""
            }

            result = app.invoke(
                initial_state,
                config={"configurable": {"thread_id": i}}
            )

            print(f"\n💡 任务分类: {result.get('task_type', 'N/A')}")

            if result.get('sql_query'):
                print(f"🔍 生成的SQL: {result['sql_query'][:100]}...")

            print(f"\n📊 查询结果摘要: ...（原始数据已保存到 last_query_result.txt）")

            print(f"\n🤖 最终回答: {result['analysis'][:200]}..." if len(
                result['analysis']) > 200 else f"\n🤖 最终回答: {result['analysis']}")

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()

    # ===== 第八步：多轮对话演示 =====
    print("\n" + "=" * 70)
    print("多轮对话演示（记忆功能）")
    print("=" * 70)

    try:
        thread_id = "demo_conversation"

        print(f"\n👤 用户: 我想了解一下这个数据库")
        state1 = app.invoke(
            {
                "messages": [HumanMessage(content="我想了解一下这个数据库")],
                "user_query": "我想了解一下这个数据库",
                "task_type": "",
                "sql_query": "",
                "query_result": "",
                "analysis": ""
            },
            config={"configurable": {"thread_id": thread_id}}
        )
        print(f"🤖 AI: {state1['analysis'][:150]}...")

        print(f"\n👤 用户: 那顾客最多的国家是哪个？")
        state2 = app.invoke(
            {
                "messages": [HumanMessage(content="那顾客最多的国家是哪个？")],
                "user_query": "那顾客最多的国家是哪个？",
                "task_type": "",
                "sql_query": "",
                "query_result": "",
                "analysis": ""
            },
            config={"configurable": {"thread_id": thread_id}}
        )
        print(f"🤖 AI: {state2['analysis'][:150]}...")

        print("\n✅ 多轮对话记忆测试成功！")

    except Exception as e:
        print(f"多轮对话失败: {e}")

    # ===== 第九步：可视化与总结 =====
    print("\n" + "=" * 70)
    print("系统总结")
    print("=" * 70)

    try:
        # 获取工作流图结构
        graph = app.get_graph()

        try:
            # 尝试新版本的方法
            graph_text = graph.draw_mermaid()
        except AttributeError:
            try:
                # 尝试旧版本的方法
                graph_text = graph.to_mermaid()
            except AttributeError:
                # 如果都不行，手动创建简单图示
                graph_text = """graph TD
    A[用户输入] --> B[智能体1-分类器]
    B -->|data_query/analysis_request| C[智能体2-SQL专家]
    B -->|general_question/schema_query| D[智能体3-分析师]
    C --> D
    D --> E[输出结果]"""

        with open("real_data_workflow.mmd", "w", encoding="utf-8") as f:
            f.write(graph_text)
        print("✓ 工作流图已保存为 real_data_workflow.mmd")

        print("\n🎯 系统架构总结:")
        print("1. 智能体1（分类器）: 分析查询意图")
        print("2. 智能体2（SQL专家）: 生成数据库查询语句")
        print("3. 智能体3（分析师）: 执行查询并分析结果")
        print("\n📊 数据处理流程:")
        print("   用户查询 → 分类 → [SQL生成] → 数据库查询 → 结果分析 → 回答")

    except Exception as e:
        print(f"可视化失败: {e}")

    print("\n" + "=" * 70)
    print("✅ 三智能体真实数据查询系统演示完成！")
    print("=" * 70)
    print("\n📁 生成的文件:")
    print("   - last_query_result.txt (最后一次查询的原始结果)")
    print("   - real_data_workflow.mmd (工作流图表)")


if __name__ == "__main__":
    # 重要：请撤销已暴露的API密钥并生成新的密钥！
    # 然后通过环境变量设置：
    # PowerShell: $env:OPENAI_API_KEY="your-new-key"
    # 或在代码中设置新的安全密钥

    # 示例（使用你自己的新密钥）：
    # os.environ["OPENAI_API_KEY"] = "your-new-openai-key"
    # os.environ["LANGSMITH_API_KEY"] = "your-new-langsmith-key"

    # 暂时禁用追踪以避免错误
    os.environ["LANGSMITH_TRACING"] = "false"

    main()