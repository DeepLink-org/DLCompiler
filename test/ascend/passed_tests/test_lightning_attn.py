import math
import pytest
import torch

from dlblas.utils.device_utils import infer_device
from dlblas.kernels.lightning_attn import lightning_attention_decode_forward
from dlblas.kernels.lightning_attn import lightning_attention_prefill_forward
from dlblas.kernels.lightning_attn import BackendType


class TestLightningAttn:

    @pytest.fixture
    def B(self, request):
        yield request.param

    @pytest.fixture
    def H(self, request):
        yield request.param

    @pytest.fixture
    def N(self, request):
        yield request.param

    @pytest.fixture
    def D(self, request):
        yield request.param

    @pytest.fixture
    def E(self, request):
        yield request.param

    @pytest.fixture
    def dtype(self, request):
        yield request.param

    @pytest.fixture
    def BLOCK_SIZE(self, request):
        yield request.param

    @pytest.fixture
    def q_states(self, B, H, N, D, dtype):
        yield torch.randn([B, H, N, D], dtype=dtype, device=infer_device())

    @pytest.fixture
    def k_states(self, B, H, N, D, dtype):
        yield torch.randn([B, H, N, D], dtype=dtype, device=infer_device())

    @pytest.fixture
    def v_states(self, B, H, N, E, dtype):
        yield torch.randn([B, H, N, E], dtype=dtype, device=infer_device())

    @pytest.fixture
    def past_key_value(self, B, H, D, E, dtype):
        yield torch.randn([B, H, D, E], dtype=dtype, device=infer_device())

    @pytest.fixture
    def slope_rate(self, H, dtype):
        def get_slopes(n):

            def get_slopes_power_of_2(n):
                start = 2**(-(2**-(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2**math.floor(math.log2(n))
                return (get_slopes_power_of_2(closest_power_of_2) + get_slopes(
                    2 * closest_power_of_2)[0::2][:n - closest_power_of_2])

        slope_rate = torch.tensor(get_slopes(H), dtype=dtype, device=infer_device()).reshape(H, 1, 1)
        yield slope_rate * (1 + 1e-5)

    # float32 only
    @pytest.mark.parametrize(
        ['B', 'H', 'N', 'D', 'E', 'dtype', 'BLOCK_SIZE'],
        [
            (1, 64, 5, 64, 128, torch.float32, 8),
            (1, 64, 72, 64, 64, torch.float32, 8),
            # (1, 64, 72, 64, 64, torch.float32, 16),
        ],
        indirect=True,
    )
    def test_lightning_attention_prefill(
        self,
        q_states,
        k_states,
        v_states,
        slope_rate,
        past_key_value,
        BLOCK_SIZE,
        dtype,
    ):
        past_key_value_torch = torch.zeros_like(past_key_value)
        past_key_value_triton = torch.zeros_like(past_key_value)
        out_torch, _ = lightning_attention_prefill_forward(q_states, k_states, v_states, past_key_value_torch, slope_rate, BLOCK_SIZE, BackendType=BackendType.TORCH)
        out_triton, _ = lightning_attention_prefill_forward(q_states, k_states, v_states, past_key_value_triton, slope_rate, BLOCK_SIZE, BackendType=BackendType.TRITON)

        if dtype == torch.float32:
            rtol=1e-03
            atol=1
        else:
            rtol=1e-03
            atol=1

        kv_check = torch.allclose(
            past_key_value_torch,
            past_key_value_triton,
            rtol=rtol,
            atol=atol,
        )
        output_check = torch.allclose(
            out_torch,
            out_triton,
            rtol=rtol,
            atol=atol,
        )

        assert kv_check, f"past_key_value torch:{past_key_value_torch}, past_key_value triton:{past_key_value_triton}"
        assert output_check, f"output torch:{out_torch}, output triton:{out_triton}"
        print(f"zmz debug torch kv_check:{past_key_value_torch}, triton :{past_key_value_triton}")

    # float32 only
    @pytest.mark.parametrize(
        ['B', 'H', 'N', 'D', 'E', 'dtype', 'BLOCK_SIZE'],
        [
            (8, 64, 1, 128, 128, torch.float32, 64),
            (16, 64, 1, 128, 128, torch.float32, 64),
        ],
        indirect=True,
    )
    def test_lightning_attention_decode(
        self,
        q_states,
        k_states,
        v_states,
        slope_rate,
        past_key_value,
        BLOCK_SIZE,
        dtype,
    ):
        past_key_value_torch = torch.zeros_like(past_key_value)
        past_key_value_triton = torch.zeros_like(past_key_value)
        out_torch, _ = lightning_attention_decode_forward(q_states, k_states, v_states, past_key_value_torch, slope_rate, BLOCK_SIZE, BackendType=BackendType.TORCH)
        out_triton, _ = lightning_attention_decode_forward(q_states, k_states, v_states, past_key_value_triton, slope_rate, BLOCK_SIZE, BackendType=BackendType.TRITON)

        if dtype == torch.float32:
            rtol=1e-03
            atol=1e-02
        else:
            rtol=1e-03
            atol=1e-02

        kv_check = torch.allclose(
            past_key_value_torch,
            past_key_value_triton,
            rtol=rtol,
            atol=atol,
        )
        output_check = torch.allclose(
            out_torch,
            out_triton,
            rtol=rtol,
            atol=atol,
        )

        assert kv_check, f"past_key_value torch:{past_key_value_torch}, past_key_value triton:{past_key_value_triton}"
        assert output_check, f"output torch:{out_torch}, output triton:{out_triton}"


if __name__ == '__main__':
    pytest.main([__file__])
