#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLIR_DIR="$SCRIPT_DIR/mlir"

# 设置 Python 环境
TRITON_PATH=$(python -c "import triton; import os; print(os.path.dirname(triton.__file__))")
export PATH="$TRITON_PATH:$TRITON_PATH/_C:$PATH"

# 检查必要工具
for tool in dicp_opt "triton-shared-opt-v3_4" FileCheck; do
    if ! command -v "$tool" &> /dev/null; then
        echo "Error: $tool is not available in PATH" >&2
        exit 1
    fi
done

# 要跳过的文件列表（只需提供文件名，不需要路径）
SKIP_FILES=(
    # 在这里添加需要跳过的文件名，例如：
    # "skip_this_test.mlir"
    # "another_skip_test.mlir"
    "linalg_broadcast.mlir"
    "linalg_multi_assign.mlir"
)

# 函数：检查文件是否应该跳过
should_skip() {
    local filename="$1"
    for skip_file in "${SKIP_FILES[@]}"; do
        if [ "$filename" = "$skip_file" ]; then
            return 0  # 应该跳过
        fi
    done
    return 1  # 不应跳过
}

# 函数：执行单个 MLIR 文件测试
run_test() {
    local mlir_file="$1"
    local filename=$(basename "$mlir_file")
    
    # 检查是否需要跳过此文件
    if should_skip "$filename"; then
        echo "SKIP: $filename - Explicitly skipped"
        return 0
    fi
    
    # 提取 RUN 指令
    local run_line=$(head -n1 "$mlir_file" | grep "^// RUN:" | sed 's|^// RUN: ||')
    if [ -z "$run_line" ]; then
        echo "SKIP: $filename - No RUN instruction found"
        return 0
    fi
    
    # 替换占位符
    local cmd=$(echo "$run_line" | sed "s|%s|$mlir_file|g" | \
                           sed 's|%dicp_opt|dicp_opt|g' | \
                           sed 's|%triton-shared-opt-v3_4|triton-shared-opt-v3_4|g' | \
                           sed 's|%FileCheck|FileCheck|g')
    
    echo "TEST: $filename"
    echo "CMD:  $cmd"
    
    # 执行测试
    if eval "$cmd"; then
        echo "PASS: $filename"
        return 0
    else
        echo "FAIL: $filename"
        return 1
    fi
}

# 处理参数：如果提供了文件名，可以跳过指定文件
if [ $# -gt 0 ]; then
    case "$1" in
        --skip)
            # 添加跳过的文件（从第二个参数开始）
            shift
            for arg in "$@"; do
                SKIP_FILES+=("$arg")
            done
            
            # 测试所有文件（包括跳过的）
            test_files=("$MLIR_DIR"/*.mlir)
            if [ ! -e "${test_files[0]}" ]; then
                echo "No MLIR files found in $MLIR_DIR" >&2
                exit 1
            fi
            ;;
        *)
            # 测试指定文件
            test_file="$1"
            if [[ "$test_file" != /* ]]; then
                test_file="$SCRIPT_DIR/$test_file"
            fi
            
            if [ ! -f "$test_file" ]; then
                echo "Error: File not found: $test_file" >&2
                exit 1
            fi
            
            run_test "$test_file"
            exit $?
            ;;
    esac
else
    # 测试所有 .mlir 文件
    test_files=("$MLIR_DIR"/*.mlir)
    if [ ! -e "${test_files[0]}" ]; then
        echo "No MLIR files found in $MLIR_DIR" >&2
        exit 1
    fi
fi

# 执行测试
total=0
passed=0
skipped=0
failed=()

for mlir_file in "${test_files[@]}"; do
    if [ -f "$mlir_file" ]; then
        ((total++))
        if run_test "$mlir_file"; then
            if should_skip "$(basename "$mlir_file")"; then
                ((skipped++))
            else
                ((passed++))
            fi
        else
            failed+=("$(basename "$mlir_file")")
        fi
    fi
done

# 输出总结
echo
echo "Results: $passed/$((total-skipped)) tests passed, $skipped skipped"
if [ $passed -lt $((total-skipped)) ]; then
    echo "Failed tests: ${failed[*]}"
    exit 1
else
    if [ ${#failed[@]} -gt 0 ]; then
        echo "Failed tests: ${failed[*]}"
        exit 1
    else
        echo "All tests passed!"
        exit 0
    fi
fi
