"""Parse ``memory_info_{aic,aiv}.json`` emitted by bishengir-compile when
invoked with ``--enable-memory-display=true``.

Each ``Record`` covers one memory scope and lists every ``memref.alloc()``
the compiler placed there. The buffer's ``extent`` field is in BITS
(verified empirically: ``extent == element_count * dtype_size_bytes * 8``).
The buffer's ``offset`` field is in BYTES.

AIC kernels expose UB as ``scope == "cbuf"``; AIV kernels use ``scope == "ub"``.
When offsets are present, the required allocation is the highest planned
end address: ``max(offset + ceil(extent_bits / 8)) * 8``. If a toolchain does
not emit offsets, fall back to a live-buffer overlap over the closed
``life_time_in_ir`` intervals.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np

UB_SCOPE_NAMES = ("cbuf", "ub")


def _tuple_or_scalar(value):
    if isinstance(value, list):
        return tuple(value)
    return (value,) if value is not None else ()


def _peak_overlap(buffers):
    """Max sum(extent) over closed live intervals via vectorized sweep.

    Emits ``(start, +extent)`` and ``(end, -extent)`` events, then
    sorts start events before end events at the same ``t``. That preserves
    closed interval semantics where ``[0, 10]`` overlaps ``[10, 20]`` at 10.
    A cumsum over deltas then yields the live allocation at each event;
    the max is the peak.
    """
    events = []
    for b in buffers:
        life = b.get("life_time_in_ir")
        extent = b.get("extent")
        if (
            isinstance(life, list)
            and len(life) == 2
            and isinstance(extent, int)
            and extent > 0
        ):
            events.append((int(life[0]), +extent))
            events.append((int(life[1]), -extent))
    if not events:
        return 0
    times, deltas = zip(*events)
    deltas = np.asarray(deltas)
    order = np.lexsort((-deltas, np.asarray(times)))
    return int(np.cumsum(deltas[order]).max())


def _required_bits_from_offsets(buffers) -> Optional[int]:
    """Return planner footprint in bits using byte offsets and bit extents."""
    max_end_bytes = 0
    found = False
    for b in buffers:
        extent = b.get("extent")
        if not isinstance(extent, int) or extent <= 0:
            continue
        extent_bytes = (extent + 7) // 8
        offsets = b.get("offset") or []
        if not isinstance(offsets, list):
            offsets = [offsets]
        for offset in offsets:
            if isinstance(offset, int) and offset >= 0:
                found = True
                max_end_bytes = max(max_end_bytes, offset + extent_bytes)
    return max_end_bytes * 8 if found else None


def peak_ub_bits(json_path: str) -> Optional[int]:
    """Required/peak UB allocation in bits, or ``None`` if no UB scope is present.

    Buffers seen in the JSON are deduplicated because some toolchains emit each
    allocation twice. ``error_info`` is intentionally ignored; the value is
    derived from structured ``memory_info_array`` fields only.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f).get("Record") or []
    peaks = []
    for rec in records:
        if (rec.get("scope") or "") not in UB_SCOPE_NAMES:
            continue
        buffers = rec.get("memory_info_array") or []
        seen = set()
        dedup = []
        for b in buffers:
            key = (
                b.get("buffer"),
                b.get("extent"),
                _tuple_or_scalar(b.get("life_time_in_ir")),
                _tuple_or_scalar(b.get("offset")),
            )
            if key in seen:
                continue
            seen.add(key)
            dedup.append(b)
        required = _required_bits_from_offsets(dedup)
        peaks.append(required if required is not None else _peak_overlap(dedup))
    return max(peaks) if peaks else None


if __name__ == "__main__":
    import sys

    bits = peak_ub_bits(sys.argv[1])
    if bits is None:
        print("no UB scope found")
        sys.exit(1)
    print(f"peak UB: {bits} bits ({bits / 8 / 1024:.1f} KiB)")
