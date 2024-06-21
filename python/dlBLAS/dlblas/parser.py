EXTRA_HASH = 'dlBLAS'


class TritonJITFunctionParser:
    '''
    parse jit'ed function under kernels/
    '''

    def __init__(self) -> None:
        pass


if __name__ == "__main__":
    import os

    # we rely on inductor's codecache
    # https://github.com/pytorch/pytorch/blob/c008488b9ce48235565825cf9e7338d72c5445d5/torch/_inductor/codecache.py#L3069
    from torch._inductor.codecache import PyCodeCache

    p = TritonJITFunctionParser()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)

    with open(os.path.join(dir_path, "kernels/persistent_matmul.py"),
              "r") as f:
        code = f.read()

    key, path = PyCodeCache.write(code, extra=EXTRA_HASH)
    mod = PyCodeCache.load_by_key_path(
        key,
        path,
    )
    print(dir(mod))
