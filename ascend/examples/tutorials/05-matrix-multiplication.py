"""
Matrix Multiplication
===============
"""
import triton
import triton.language as tl
import torch
import torch_npu
import pytest

def torch_dot_None(x0, x1):
    res = torch.matmul(x0, x1)
    return res

def generate_tensor(shape, dtype):
    if dtype == 'float32' or dtype == 'float16' or dtype == 'bfloat16':
        return torch.randn(size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16':
        return torch.randint(low=0, high=2000, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int8':
        return torch.randint(low=0, high=127, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'bool':
        return torch.randint(low=0, high=2, size=shape).bool()
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))

def validate_cmp(dtype, y_cal, y_ref):
    y_cal = y_cal.npu()
    y_ref = y_ref.npu()
    if dtype == 'float16':
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype == 'bfloat16':
        torch.testing.assert_close(y_ref.to(torch.float32), y_cal.to(torch.float32), rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype == 'float32':
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-04, atol=1e-04, equal_nan=True)
    elif dtype == 'int32' or dtype == 'int64'or dtype == 'int16' or dtype == 'int8':
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'bool':
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))

def get_torch_typename(dtype):
    if dtype == 'float32':
        tyname = torch.float32
    elif dtype == 'int32':
        tyname = torch.int32
    elif dtype == 'int64':
        tyname = torch.int64
    elif dtype == 'float16':
        tyname = torch.float16
    elif dtype == 'int16':
        tyname = torch.int16
    elif dtype == 'int8':
        tyname = torch.int8
    elif dtype == 'bool':
        tyname = torch.bool
    elif dtype == 'bfloat16':
        tyname = torch.bfloat16
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))
    return tyname

@triton.jit
def triton_dot_2_None(output_ptr, x_ptr, y_ptr, z_ptr,A : tl.constexpr,B : tl.constexpr,C : tl.constexpr,D : tl.constexpr):
    aidx=tl.arange(0,A)
    bidx=tl.arange(0,B)
    cidx=tl.arange(0,C)
    didx=tl.arange(0,D)
    accumulator = tl.zeros((B, D), dtype=tl.float32)
    Xidx=bidx[:,None]*C+cidx[None,:]
    Yidx=cidx[:,None]*D+didx[None,:]
    Zidx=bidx[:,None]*D+didx[None,:]
    X = tl.load(x_ptr+Xidx)
    Y = tl.load(y_ptr+Yidx)
    Z = tl.load(z_ptr+Zidx)
    ret = tl.dot(X, Y)
    oidx=bidx[:,None]*D+didx[None,:]
    tl.store(output_ptr+oidx,ret)
    
testlist = [
    (3, 16, 16, 16),
]
typelist = ['float16',]

@pytest.mark.parametrize('A, B, C, D',testlist)
@pytest.mark.parametrize('sigtype',typelist)
def test_dot_2_None(sigtype, A, B, C, D):
    dtype = get_torch_typename(sigtype)
    x0 = generate_tensor(shape = (B, C),dtype = sigtype).npu()
    x1 = generate_tensor(shape = (C, D),dtype = sigtype).npu()
    if 'int' in sigtype:
        x2 = generate_tensor(shape = (B, D),dtype = 'int32').npu()
        ans = torch_dot_None(x0.to(torch.float32), x1.to(torch.float32)).to(dtype)
    else:
        x2 = generate_tensor(shape = (B, D),dtype = 'float32').npu()
        ans = torch_dot_None(x0, x1)
    output = torch.zeros((B, D), dtype = dtype).npu()
    triton_dot_2_None[1,1,1](output, x0, x1, x2, A, B, C, D, debug = True)
    validate_cmp(sigtype,output,ans)

if __name__=='__main__':
    for A,B,C,D in testlist:
        for sigtype in typelist:
            try:
                test_dot_2_None(sigtype, A, B, C, D)
                print(f"Test matmul with dtype={sigtype}, shape=({A},{B},{C},{D}) PASSED!")
            except AssertionError as e:
                print(f"Test matmul with dtype={sigtype}, shape=({A},{B},{C},{D}) FAILED!")