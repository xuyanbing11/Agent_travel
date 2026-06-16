# test_imports.py
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

import langchain
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

print("✅ 所有包导入成功！")
print(f"SQLAlchemy 版本：{sqlalchemy.__version__}")
print(f"LangChain 版本：{langchain.__version__}")