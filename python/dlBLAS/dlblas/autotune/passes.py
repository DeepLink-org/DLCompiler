import os
import re


def find_call_pattern_index_from_text(text: str,
                                      pattern: str) -> list[tuple[int]]:
    '''call pattern:
        xxx(whatever)
        |           |
       / \         / \
      start_idx   end_idx

        1. find `xxx` based on pattern
        2. find the first `(`
        3. find the last `)`
    '''
    start_idx = []
    matches: list[re.Match] = re.finditer(
        pattern,
        text,
        flags=re.MULTILINE,
    )
    for match in matches:
        start_idx.append(match.start())

    start_end_idx = []
    for start in start_idx:
        # goes to the first '('
        end = start
        while True:
            if text[end] == '(':
                break
            end += 1

        # find the last closing ')'
        # there must be one, otherwise the file will report error at import time
        open_count = 1
        end += 1
        while True:
            if text[end] == '(':
                open_count += 1
            elif text[end] == ')':
                open_count -= 1
                if open_count == 0:
                    break
            end += 1
        end += 1
        start_end_idx.append((start, end))
    return start_end_idx


# ======================
# transform pass
# ======================
def rewrite_dlblas_registration_pass(text: str) -> str:
    '''rewrite all `register_dlblas_op()` call -> pass
    '''
    # find pattern
    register_pattern = r'register_dlblas_op\(.*?(?=\n|$)'
    start_end_idx = find_call_pattern_index_from_text(text, register_pattern)

    # build new src code
    new_src = ''
    last_end = 0
    for start, end in start_end_idx:
        new_src += text[last_end:start]
        new_src += 'pass'
        last_end = end

    # the remaining part
    new_src += text[last_end:]
    return new_src


# ======================
# analysis pass
# ======================
def analyse_kernel_call_pass(text: str, kernel_name: str) -> list[tuple[int]]:
    '''find invoke kernel idx in the src text file; 
    
    Triton kernel call have this pattern: {kernel_name}[{grid_name}]
    '''
    kernel_call_pattern = fr'{kernel_name}\[[a-zA-Z0-9_]+\]'
    start_end_idx = find_call_pattern_index_from_text(
        text,
        kernel_call_pattern,
    )
    return start_end_idx
