#!/bin/bash
export LANG="zh_CN.UTF-8"
export LC_ALL="zh_CN.UTF-8"

modified_py_files=$(git diff --name-only origin/main...HEAD | grep '\.py$')
if [ -z "$modified_py_files" ]; then
 echo "No Python files modified in this PR, skipping Black formatting"
 exit 0
fi

echo "Found modified Python files for PR:"
echo "$modified_py_files" | tr '\n' ' '  # 显示修改的文件列表

# 使用 Black 检查这些文件（只检查，不修改）
pip install black
black --check --diff $modified_py_files