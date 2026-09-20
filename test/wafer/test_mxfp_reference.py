"""Check the independent scaled-dot oracle against fixed format encodings."""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
path = Path(__file__).resolve().parents[2] / "third_party/wafer/examples/_wafer_reference.py"
spec = importlib.util.spec_from_file_location("wafer_mxfp_reference", path)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mxfp_known_encodings_and_scales(dtype):
    # Both nibble order and all signed FP4 values, including signed zero.
    packed = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2], dtype=torch.uint8)
    expected = torch.tensor([[0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6] * 2], dtype=dtype)
    one = torch.tensor([[127]], dtype=torch.uint8)
    actual = reference.upcast_mxfp_cpu(packed, one, "e2m1", dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.signbit(actual), torch.signbit(expected))
    formats = [
        ("e4m3", [0, 1, 0x38, 0x40, 0x7E, 0x7F, 0x80, 0xB8],
         [0, 2**-9, 1, 2, 448, float("nan"), -0., -1]),
        ("e5m2", [0, 1, 0x3C, 0x40, 0x7B, 0x7C, 0x7F, 0xBC],
         [0, 2**-16, 1, 2, 57344, float("inf"), float("nan"), -1]),
    ]
    for name, codes, values in formats:
        packed = torch.tensor([codes * 4], dtype=torch.uint8)
        actual = reference.upcast_mxfp_cpu(packed, one, name, dtype)
        torch.testing.assert_close(actual, torch.tensor([values * 4], dtype=dtype),
                                   rtol=0, atol=0, equal_nan=True)
    for scale, value in [(0, 0), (126, .5), (127, 1), (128, 2), (255, float("nan"))]:
        actual = reference.upcast_mxfp_cpu(
            torch.full((1, 16), 0x22, dtype=torch.uint8),
            torch.tensor([[scale]], dtype=torch.uint8), "e2m1", dtype)
        torch.testing.assert_close(actual, torch.full((1, 32), value, dtype=dtype),
                                   rtol=0, atol=0, equal_nan=True)
