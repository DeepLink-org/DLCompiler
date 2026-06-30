import json
import tempfile

from backend.ascend_autotune_runtime.resource_memory_parser import peak_ub_bits


def _make_json(records):
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump({"Header": {"KernelName": "k"}, "Record": records}, f)
    f.close()
    return f.name


def test_partial_overlap_peak_at_intersection():
    p = _make_json(
        [
            {
                "scope": "cbuf",
                "memory_info_array": [
                    {"extent": 1000, "life_time_in_ir": [0, 10], "buffer": "a"},
                    {"extent": 2000, "life_time_in_ir": [5, 15], "buffer": "b"},
                ],
            }
        ]
    )
    assert peak_ub_bits(p) == 3000


def test_dedupe_identical_buffers():
    """Some toolchains emit each allocation twice; count once."""
    p = _make_json(
        [
            {
                "scope": "cbuf",
                "memory_info_array": [
                    {"extent": 1000, "life_time_in_ir": [0, 10], "buffer": "x"},
                    {"extent": 1000, "life_time_in_ir": [0, 10], "buffer": "x"},
                ],
            }
        ]
    )
    assert peak_ub_bits(p) == 1000


def test_planner_offsets_define_required_bits_and_ignore_error_info():
    p = _make_json(
        [
            {
                "scope": "ub",
                "status": "fail",
                "error_info": "ub overflow, requires 999999 bits while 1572864 bits available!",
                "memory_info_array": [
                    {
                        "extent": 262144,
                        "offset": [296192],
                        "life_time_in_ir": [57, 132],
                        "buffer": "a",
                    },
                ],
            }
        ]
    )
    assert peak_ub_bits(p) == 2631680


def test_returns_none_when_no_ub():
    p = _make_json(
        [
            {
                "scope": "cc",
                "memory_info_array": [
                    {"extent": 1000, "life_time_in_ir": [0, 10], "buffer": "x"},
                ],
            }
        ]
    )
    assert peak_ub_bits(p) is None
