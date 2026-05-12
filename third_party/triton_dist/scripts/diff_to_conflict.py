################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
#!/usr/bin/env python3
"""
Generate a Git-style conflict file from two input files.
Usage: python3 diff_to_conflict.py <file1> <file2>

This script creates a merged file with Git conflict markers (<<<<<<<, =======, >>>>>>>)
where the two input files differ, allowing manual conflict resolution in editors like VS Code.
"""

import sys
import difflib


def read_file(filename):
    """Read file contents and return as list of lines."""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.readlines()
    except Exception as e:
        sys.stderr.write(f"Error reading {filename}: {str(e)}\n")
        sys.exit(1)


def generate_conflict_file(file1_path, file2_path):
    """Generate a Git-style conflict file from two input files."""
    # Read files
    file1_lines = read_file(file1_path)
    file2_lines = read_file(file2_path)

    # Create a sequence matcher to identify differences
    matcher = difflib.SequenceMatcher(None, file1_lines, file2_lines)

    # Process the differences and generate conflict markers
    result = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # No conflict, just copy the matching lines
            result.extend(file1_lines[i1:i2])
        elif tag == 'replace' or tag == 'delete' or tag == 'insert':
            # There's a conflict
            # First part (file1)
            result.append(f"<<<<<<< {file1_path}\n")
            if i1 < i2:  # There's content in file1
                result.extend(file1_lines[i1:i2])

            # Separator
            result.append("=======\n")

            # Second part (file2)
            if j1 < j2:  # There's content in file2
                result.extend(file2_lines[j1:j2])

            # End marker
            result.append(f">>>>>>> {file2_path}\n")

    # Output the result
    sys.stdout.write(''.join(result))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write(f"Usage: {sys.argv[0]} <file1> <file2>\n")
        sys.exit(1)

    file1_path = sys.argv[1]
    file2_path = sys.argv[2]

    generate_conflict_file(file1_path, file2_path)
