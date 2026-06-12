#!/bin/bash
export LANG="zh_CN.UTF-8"
export LC_ALL="zh_CN.UTF-8"

modified_py_files=$(git diff --name-only --diff-filter=ACMR origin/main...HEAD | grep '\.py$')
if [ -z "$modified_py_files" ]; then
 echo "No Python files modified in this PR, skipping Black formatting"
 exit 0
fi

echo "Found modified Python files for PR:"
echo "$modified_py_files" | tr '\n' ' '  # 显示修改的文件列表

pip install  "black==25.*"
black --check --diff $modified_py_files