"""Wafer must retain stable scalar libm log1p calls through its actual pipeline."""
import pytest


@pytest.mark.parametrize("dtype,symbol", [("f32", "log1pf"), ("f64", "log1p")])
def test_stable_log1p_lowering(dtype, symbol, wafer_modules, monkeypatch):
    from triton.backends.dicp_triton.wafer import _find_wafer_opt

    _, compiler, _ = wafer_modules
    # Exercise the source pipeline with the installed, versioned compiler tool.
    monkeypatch.setattr(compiler, "_find_wafer_opt", _find_wafer_opt)
    ir = f"""module {{
      func.func @stable_log1p(%x: {dtype}) -> {dtype} {{
        %y = math.log1p %x : {dtype}
        return %y : {dtype}
      }}
    }}"""
    llvm = compiler.wafer_ir_to_llir(ir, {})
    assert f"@{symbol}(" in llvm
    assert "@llvm.log." not in llvm
    assert "fadd" not in llvm
