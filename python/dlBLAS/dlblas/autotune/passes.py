import os
import re


# ======================
# transform pass
# ======================
def rewrite_dlblas_registeration_pass(text: str) -> str:
    # FIXME multiple line of args
    register_pat = r'register_dlblas_op\(.*?(?=\n|$)'

    def rewrite_register(match: re.Match):
        return 'pass\n'

    # replace all `register_dlblas_op` call
    replaced_text = re.sub(
        register_pat,
        rewrite_register,
        text,
        flags=re.MULTILINE,
    )

    return replaced_text


# ======================
# analysis pass
# ======================
def analyse_kernel_call_pass(text: str, kernel_name: str) -> list[tuple[int]]:
    '''find invoke kernel idx in the src text file; 
    
    Triton kernel call have this pattern: {kernel_name}[{grid_name}]
    '''
    start_idx = []
    invoke_kernel_pattern = fr'{kernel_name}\[[a-zA-Z0-9_]+\]'
    matches: list[re.Match] = re.finditer(
        invoke_kernel_pattern,
        text,
        flags=re.DOTALL,
    )
    for match in matches:
        start_idx.append(match.start())

    # find (start, end) pair in the kernel text
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
