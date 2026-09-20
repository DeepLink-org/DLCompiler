# flagtree tle
import triton.language.core as tl
import builtins
import math
import triton
import triton.language as language


def _tile_offsets(x, index, tile_shape, semantic):
    """Normalize FlagTree's row-major tile indices to 3.5 DSA slice offsets."""
    shape = tuple(tl._unwrap_if_constexpr(dim) for dim in x.shape)
    tile_shape = tuple(tl._unwrap_if_constexpr(dim) for dim in tl._unwrap_if_constexpr(tile_shape))
    if len(shape) != len(tile_shape) or any(type(t) is not int or t <= 0 for t in tile_shape):
        raise ValueError("tile_shape must contain positive integers and match source rank")
    if any(s % t for s, t in zip(shape, tile_shape)):
        raise ValueError("source dimensions must be divisible by tile dimensions")
    grid = tuple(s // t for s, t in zip(shape, tile_shape))
    index = tl._unwrap_if_constexpr(index)
    if isinstance(index, (tuple, list, tl.tuple)):
        coords = [tl._unwrap_if_constexpr(i) for i in index]
        if len(coords) != len(grid):
            raise ValueError("tile index rank must match source rank")
    else:
        if isinstance(index, tl.tensor):
            if len(index.shape) or not index.dtype.is_int():
                raise ValueError("dynamic tile index must be a scalar integer")
        elif type(index) is not int or not 0 <= index < math.prod(grid):
            raise ValueError("linear tile index out of range")
        coords = [None] * len(grid)
        for axis in builtins.range(len(grid) - 1, -1, -1):
            if isinstance(index, tl.tensor):
                coords[axis] = index.__mod__(grid[axis], _semantic=semantic)
                index = index.__floordiv__(grid[axis], _semantic=semantic)
            else:
                coords[axis], index = index % grid[axis], index // grid[axis]
    offsets = []
    for coord, extent, tile in zip(coords, grid, tile_shape):
        if isinstance(coord, tl.tensor):
            if len(coord.shape) or not coord.dtype.is_int():
                raise ValueError("dynamic tile indices must be scalar integers")
            offsets.append(coord.__mul__(tile, _semantic=semantic))
        else:
            if type(coord) is not int or not 0 <= coord < extent:
                raise ValueError("tile index out of range")
            offsets.append(coord * tile)
    return offsets, tile_shape


@tl.builtin
def extract_tile(x, index, tile_shape, _semantic=None):
    """Extract a tile using a scalar or per-axis row-major tile index.

    Dynamic indices must be in bounds. The official 3.5 integration lowers
    directly through DSA slices rather than requiring FlagTree's shared IR.
    """
    from .dsa import extract_slice
    offsets, shape = _tile_offsets(x, index, tile_shape, _semantic)
    return extract_slice(x, offsets, shape, (1,) * len(shape), _semantic=_semantic)


@tl.builtin
def insert_tile(x, tile, index, _semantic=None):
    """Insert a tile using a scalar or per-axis row-major tile index."""
    from .dsa import insert_slice
    offsets, shape = _tile_offsets(x, index, tile.shape, _semantic)
    return insert_slice(x, tile, offsets, shape, (1,) * len(shape), _semantic=_semantic)


@triton.jit
def cumsum(x, axis: tl.constexpr = 0, reverse: tl.constexpr = False):
    """Exclusive rank-one float scan and total, using the existing Wafer lowering.

    Shift the input before scanning to avoid cancellation in inclusive_sum - x.
    The JIT wrapper lets Triton 3.5 inline its standard scan/reduce functions.
    """
    tl.static_assert(len(x.shape) == 1 and (axis == 0 or axis == -1) and not reverse,
                     "Wafer TLE cumsum supports rank-one forward scans")
    tl.static_assert(x.dtype.is_floating(), "Wafer TLE cumsum requires floating point")
    indices = tl.arange(0, x.shape[0])
    previous = tl.maximum(indices - 1, 0)
    shifted = tl.gather(x, previous, axis=0)
    shifted = tl.where(indices > 0, shifted, 0)
    return language.cumsum(shifted, 0), language.sum(x, 0)


# -----------------------
# Non-Atomic Memory Operations
# -----------------------


@tl.builtin
def load(pointer, mask=None, other=None, boundary_check=(), padding_option="", cache_modifier="", eviction_policy="",
         volatile=False, is_async=False, _semantic=None):
    """
    Return a tensor of data whose values are loaded from memory at location defined by `pointer`:

        (1) If `pointer` is a single element pointer, a scalar is be loaded.  In
            this case:

            - `mask` and `other` must also be scalars,
            - `other` is implicitly typecast to `pointer.dtype.element_ty`, and
            - `boundary_check` and `padding_option` must be empty.

        (2) If `pointer` is an N-dimensional tensor of pointers, an
            N-dimensional tensor is loaded.  In this case:

            - `mask` and `other` are implicitly broadcast to `pointer.shape`,
            - `other` is implicitly typecast to `pointer.dtype.element_ty`, and
            - `boundary_check` and `padding_option` must be empty.

        (3) If `pointer` is a block pointer defined by `make_block_ptr`, a
            tensor is loaded.  In this case:

            - `mask` and `other` must be `None`, and
            - `boundary_check` and `padding_option` can be specified to control the behavior of out-of-bound access.

    :param pointer: Pointer to the data to be loaded
    :type pointer: `triton.PointerType`, or block of `dtype=triton.PointerType`
    :param mask: if `mask[idx]` is false, do not load the data at address `pointer[idx]`
        (must be `None` with block pointers)
    :type mask: Block of `triton.int1`, optional
    :param other: if `mask[idx]` is false, return `other[idx]`
    :type other: Block, optional
    :param boundary_check: tuple of integers, indicating the dimensions which should do the boundary check
    :type boundary_check: tuple of ints, optional
    :param padding_option: should be one of {"", "zero", "nan"}, the padding value to use while out of bounds. "" means an undefined value.
    :param cache_modifier: changes cache option in NVIDIA PTX
    :type cache_modifier: str, optional, should be one of {"", ".ca", ".cg", ".cv"}, where ".ca" stands for
        cache at all levels, ".cg" stands for cache at global level (cache in L2 and below, not L1),
        and ".cv" means don’t cache and fetch again. see
        `cache operator <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators>`_ for more details.
    :param eviction_policy: changes eviction policy in NVIDIA PTX
    :type eviction_policy: str, optional
    :param volatile: changes volatile option in NVIDIA PTX
    :type volatile: bool, optional
    """
    x = tl.load(pointer, mask=mask, other=other, boundary_check=boundary_check, padding_option=padding_option,
                cache_modifier=cache_modifier, eviction_policy=eviction_policy, volatile=volatile, _semantic=_semantic)
    x.handle.set_attr("tt.load.async", _semantic.builder.get_bool_attr(is_async))
    return x
