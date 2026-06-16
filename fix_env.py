# fix_env.py
import os
import sys
import subprocess
import shutil

print("=== 环境诊断 ===")

# 1. 检查 Python 路径
python_exe = sys.executable
print(f"Python: {python_exe}")

# 2. 检查是否在 conda 环境中
conda_prefix = os.getenv('CONDA_PREFIX', '')
print(f"Conda 环境: {conda_prefix}")

# 3. 检查 Desktop 文件
desktop = os.path.join(os.path.expanduser('~'), 'Desktop', 'info')
if os.path.exists(desktop):
    print(f"⚠️ 找到 Desktop/info 文件: {desktop}")
    try:
        os.remove(desktop)
        print("✅ 已删除")
    except:
        print("❌ 无法删除")
else:
    print("✅ Desktop/info 不存在")

# 4. 建议
print("\n=== 建议 ===")
if 'conda' in python_exe.lower():
    print("使用 Conda 环境，建议重新创建环境")
else:
    print("使用系统 Python，建议检查 PATH")