"""Opt-in static bounded tensor slices using Wafer's TLE slice operation."""

import builtins

import triton.language.core as tl
from triton.experimental.tle.language.dsa import extract_slice


_upstream_getitem = tl.tensor.__getitem__


@tl._tensor_member_fn
@tl.builtin
def __getitem__(self, slices, _semantic=None):
    if isinstance(slices, tl.tuple):
        indices = list(slices.values)
    elif isinstance(slices, (builtins.tuple, list)):
        indices = list(slices)
    else:
        indices = [slices]
    slice_types = (builtins.slice, tl.slice)
    bounded = any(isinstance(s, slice_types) and any(
        tl._unwrap_if_constexpr(v) is not None for v in (s.start, s.stop, s.step)
    ) for s in indices)
    if not bounded:
        # Preserve upstream full-slice and new-axis behavior.
        return _upstream_getitem(self, slices, _semantic=_semantic)
    if len(indices) > len(self.shape) or not all(isinstance(s, slice_types) for s in indices):
        raise ValueError("Wafer bounded indexing accepts slices only; use expand_dims separately")
    indices += [builtins.slice(None)] * (len(self.shape) - len(indices))
    offsets, sizes, strides = [], [], []
    for index, dim in zip(indices, self.type.shape):
        values = [tl._unwrap_if_constexpr(v) for v in (index.start, index.stop, index.step)]
        start, stop, step = [default if v is None else v for v, default in zip(values, (0, dim, 1))]
        if not all(isinstance(v, int) for v in (start, stop, step)):
            raise ValueError("Wafer slice bounds must be compile-time integers")
        if not 0 <= start < stop <= dim or step <= 0:
            raise ValueError("Wafer slices require 0 <= start < stop <= dimension and positive stride")
        offsets.append(start)
        sizes.append((stop - start + step - 1) // step)
        strides.append(step)
    return extract_slice(self, offsets, sizes, strides, _semantic=_semantic)
