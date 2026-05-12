from __future__ import annotations

from functools import reduce
from typing import Any, Callable, TYPE_CHECKING, Union, List, Dict

if TYPE_CHECKING:
    from .language import core
    IterableType = Union[list[Any], tuple[Any, ...], core.tuple, core.tuple_type]
    ObjPath = tuple[int, ...]

TRITON_MAX_TENSOR_NUMEL = 1048576


def get_iterable_path(iterable: IterableType, path: ObjPath) -> Any:
    return reduce(lambda a, idx: a[idx], path, iterable)  # type: ignore[index]


def set_iterable_path(iterable: IterableType, path: tuple[int, ...], val: Any):
    from .language import core
    assert len(path) != 0
    prev = iterable if len(path) == 1 else get_iterable_path(iterable, path[:-1])
    assert isinstance(prev, core.tuple)
    prev._setitem(path[-1], val)


def find_paths_if(iterable: Union[IterableType, Any], pred: Callable[[ObjPath, Any], bool]) -> list[ObjPath]:
    from .language import core
    is_iterable: Callable[[Any], bool] = lambda x: isinstance(x, (list, tuple, core.tuple, core.tuple_type))
    # We need to use dict so that ordering is maintained, while set doesn't guarantee order
    ret: dict[ObjPath, None] = {}

    def _impl(path: tuple[int, ...], current: Any):
        if is_iterable(current):
            for idx, item in enumerate(current):
                _impl((*path, idx), item)
        elif pred(path, current):
            ret[path] = None

    _impl((), iterable)

    return list(ret.keys())


def is_power_of_two(x):
    return (x & (x - 1)) == 0


def validate_block_shape(shape: List[int]):
    numel = 1
    for i, d in enumerate(shape):
        if not isinstance(d, int):
            raise TypeError(f"Shape element {i} must have type `constexpr[int]`, got `constexpr[{type(d)}]")
        if not is_power_of_two(d):
            raise ValueError(f"Shape element {i} must be a power of 2")
        numel *= d

    if numel > TRITON_MAX_TENSOR_NUMEL:
        raise ValueError(f"numel ({numel}) exceeds triton maximum tensor numel ({TRITON_MAX_TENSOR_NUMEL})")
    return numel


type_canonicalisation_dict = {
    # we canonicalise all bools to be unsigned:
    "bool": "u1",
    "int1": "u1",
    "uint1": "u1",
    "i1": "u1",
    # floating-point dtypes:
    "float8e4nv": "fp8e4nv",
    "float8e5": "fp8e5",
    "float8e4b15": "fp8e4b15",
    "float8_e4m3fn": "fp8e4nv",
    "float8e4b8": "fp8e4b8",
    "float8_e4m3fnuz": "fp8e4b8",
    "float8_e5m2": "fp8e5",
    "float8e5b16": "fp8e5b16",
    "float8_e5m2fnuz": "fp8e5b16",
    "half": "fp16",
    "float16": "fp16",
    "bfloat16": "bf16",
    "float": "fp32",
    "float32": "fp32",
    "double": "fp64",
    "float64": "fp64",
    # signed integers:
    "int8": "i8",
    "int16": "i16",
    "int": "i32",
    "int32": "i32",
    "int64": "i64",
    # unsigned integers:
    "uint8": "u8",
    "uint16": "u16",
    "uint32": "u32",
    "uint64": "u64",
    "void": "void",
}

for v in list(type_canonicalisation_dict.values()):
    type_canonicalisation_dict[v] = v


def canonicalize_dtype(dtype):
    dtype_str = str(dtype).split(".")[-1]
    return type_canonicalisation_dict[dtype_str]


def canonicalize_ptr_dtype(dtype, is_const):
    return f"{'*k' if is_const else '*'}{canonicalize_dtype(dtype)}"


BITWIDTH_DICT: Dict[str, int] = {
    **{f"u{n}": n
       for n in (1, 8, 16, 32, 64)},
    **{f"i{n}": n
       for n in (1, 8, 16, 32, 64)},
    **{f"fp{n}": n
       for n in (16, 32, 64)},
    **{f"fp8{suffix}": 8
       for suffix in ("e4nv", "e4b15", "e4b8", "e5", "e5b16")},
    "bf16": 16,
    "void": 0,
}

for k, v in type_canonicalisation_dict.items():
    BITWIDTH_DICT[k] = BITWIDTH_DICT[v]


def get_primitive_bitwidth(dtype: str) -> int:
    return BITWIDTH_DICT[dtype]


def is_namedtuple(val):
    return isinstance(val, type) and issubclass(val, tuple) and hasattr(val, "_fields")


def print_matrix(matrix, title=""):
    """Helper function to print a 2D matrix nicely.
    E.g.:
    import triton
    from triton.tools import LinearLayout
    layout = LinearLayout.from_bases(
        [
            ("register", [[0, 1], [1, 0]]),                           # 4 registers
            ("lane", [[0, 2], [0, 4], [0, 8], [2, 0], [4, 0]]),       # 32 lanes
            ("warp", [[8, 0]]),                                       # 2 warps
        ],
        ["row", "col"],
    )
    matrix = layout.get_2d_matrix_view(row_col_dims=("row", "col"), print_info=(0,1,0,0))
    triton._utils.print_matrix(matrix)

    >>      Col 0 Col 1 Col 2 Col 3 Col 4 Col 5 Col 6 Col 7 Col 8 Col 9 Col10 Col11 Col12 Col13 Col14 Col15
    >>      -----------------------------------------------------------------------------------------------
    >> Row 0 |  t0 |  t0 |  t1 |  t1 |  t2 |  t2 |  t3 |  t3 |  t4 |  t4 |  t5 |  t5 |  t6 |  t6 |  t7 |  t7 |
    >> Row 1 |  t0 |  t0 |  t1 |  t1 |  t2 |  t2 |  t3 |  t3 |  t4 |  t4 |  t5 |  t5 |  t6 |  t6 |  t7 |  t7 |
    >> Row 2 |  t8 |  t8 |  t9 |  t9 | t10 | t10 | t11 | t11 | t12 | t12 | t13 | t13 | t14 | t14 | t15 | t15 |
    >> Row 3 |  t8 |  t8 |  t9 |  t9 | t10 | t10 | t11 | t11 | t12 | t12 | t13 | t13 | t14 | t14 | t15 | t15 |
    >> Row 4 | t16 | t16 | t17 | t17 | t18 | t18 | t19 | t19 | t20 | t20 | t21 | t21 | t22 | t22 | t23 | t23 |
    >> Row 5 | t16 | t16 | t17 | t17 | t18 | t18 | t19 | t19 | t20 | t20 | t21 | t21 | t22 | t22 | t23 | t23 |
    >> Row 6 | t24 | t24 | t25 | t25 | t26 | t26 | t27 | t27 | t28 | t28 | t29 | t29 | t30 | t30 | t31 | t31 |
    >> Row 7 | t24 | t24 | t25 | t25 | t26 | t26 | t27 | t27 | t28 | t28 | t29 | t29 | t30 | t30 | t31 | t31 |
    >> Row 8 |  t0 |  t0 |  t1 |  t1 |  t2 |  t2 |  t3 |  t3 |  t4 |  t4 |  t5 |  t5 |  t6 |  t6 |  t7 |  t7 |
    >> Row 9 |  t0 |  t0 |  t1 |  t1 |  t2 |  t2 |  t3 |  t3 |  t4 |  t4 |  t5 |  t5 |  t6 |  t6 |  t7 |  t7 |
    >> Row10 |  t8 |  t8 |  t9 |  t9 | t10 | t10 | t11 | t11 | t12 | t12 | t13 | t13 | t14 | t14 | t15 | t15 |
    >> Row11 |  t8 |  t8 |  t9 |  t9 | t10 | t10 | t11 | t11 | t12 | t12 | t13 | t13 | t14 | t14 | t15 | t15 |
    >> Row12 | t16 | t16 | t17 | t17 | t18 | t18 | t19 | t19 | t20 | t20 | t21 | t21 | t22 | t22 | t23 | t23 |
    >> Row13 | t16 | t16 | t17 | t17 | t18 | t18 | t19 | t19 | t20 | t20 | t21 | t21 | t22 | t22 | t23 | t23 |
    >> Row14 | t24 | t24 | t25 | t25 | t26 | t26 | t27 | t27 | t28 | t28 | t29 | t29 | t30 | t30 | t31 | t31 |
    >> Row15 | t24 | t24 | t25 | t25 | t26 | t26 | t27 | t27 | t28 | t28 | t29 | t29 | t30 | t30 | t31 | t31 |
    """
    if title:
        print(f"\n{title}")
        print("=" * len(title))

    if not matrix:
        print("(empty matrix)")
        return

    # Calculate column widths for alignment
    col_widths = []
    for col_idx in range(len(matrix[0])):
        max_width = max(len(str(row[col_idx])) for row in matrix)
        col_widths.append(max_width + 2)

    # Print header
    header = "     " + " ".join(f"Col{i:2d}".center(col_widths[i]) for i in range(len(matrix[0])))
    print(header)
    print("     " + "-" * (sum(col_widths) + len(matrix[0]) - 1))

    # Print each row
    for row_idx, row in enumerate(matrix):
        cells = [str(cell).center(col_widths[i]) for i, cell in enumerate(row)]
        print(f"Row{row_idx:2d} |" + "|".join(cells) + "|")

    print()
