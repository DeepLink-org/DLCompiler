#!/bin/bash

set -e

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

cd "$script_dir"
find tools/ compiler/ -regex '.*\.\(h\|cpp\|cc\|c\)' -print0 | xargs -0 clang-format -i
