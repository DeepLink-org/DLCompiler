#!/bin/bash

set -e

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

cd "$script_dir"

# Determine the base ref: if on a branch with origin/main, diff against main; otherwise diff against HEAD~1
if git rev-parse --verify origin/main &> /dev/null; then
  base_ref="origin/main"
else
  base_ref="HEAD~1"
fi

# Collect changed files, excluding third_party and .git (per format.yaml ignore rules)
# Include: committed changes vs base, staged changes, and unstaged working tree changes
filter_exclude() {
  grep -v '^third_party/' | grep -v '^\.git/' || true
}

changed_cpp_files=$({
  git diff --name-only --diff-filter=ACMR "$base_ref"...HEAD
  git diff --cached --name-only --diff-filter=ACMR
  git diff --name-only --diff-filter=ACMR
} | sort -u | grep -E '\.(h|cpp|cc|c)$' | filter_exclude)
changed_md_files=$({
  git diff --name-only --diff-filter=ACMR "$base_ref"...HEAD
  git diff --cached --name-only --diff-filter=ACMR
  git diff --name-only --diff-filter=ACMR
} | sort -u | grep '\.md$' | filter_exclude)
changed_py_files=$({
  git diff --name-only --diff-filter=ACMR "$base_ref"...HEAD
  git diff --cached --name-only --diff-filter=ACMR
  git diff --name-only --diff-filter=ACMR
} | sort -u | grep '\.py$' | filter_exclude)

# clang-format
if [ -n "$changed_cpp_files" ]; then
  if command -v clang-format &> /dev/null; then
    echo "clang-format on changed C/C++ files:"
    echo "$changed_cpp_files"
    echo "$changed_cpp_files" | xargs clang-format -i
  else
    echo "clang-format not found, skipping C/C++ formatting"
  fi
else
  echo "No C/C++ files modified, skipping clang-format"
fi

# markdownlint
if [ -n "$changed_md_files" ]; then
  if command -v markdownlint-cli2 &> /dev/null; then
    echo "markdownlint on changed Markdown files:"
    echo "$changed_md_files"
    echo "$changed_md_files" | xargs markdownlint-cli2 --fix
  else
    echo "markdownlint-cli2 not found, skipping markdown linting"
  fi
else
  echo "No Markdown files modified, skipping markdownlint"
fi

# python-black
if [ -n "$changed_py_files" ]; then
  if command -v black &> /dev/null; then
    echo "black on changed Python files:"
    echo "$changed_py_files"
    echo "$changed_py_files" | xargs black
  else
    echo "black not found, skipping python formatting"
  fi
else
  echo "No Python files modified, skipping black"
fi
