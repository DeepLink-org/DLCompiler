# For https://github.com/InternLM/lmdeploy/tree/v0.6.1
import torch
from dlblas.kernels.fill_kv_cache import fill_kv_cache
from dlblas.kernels.apply_rotary_pos_emb import apply_rotary_pos_emb
from dlblas.kernels.paged_attention import paged_attention_fwd
from dlblas.kernels.rms_norm import rms_norm
from dlblas.kernels.multinomial_sampling import multinomial_sampling
from dlblas.kernels.activation import silu_and_mul


def _patch_lmdeploy():
    import lmdeploy.pytorch.kernels.cuda as lmdeploy_kernels

    DEFAULT_PATCH_LIST = [
        "fill_kv_cache",
        "apply_rotary_pos_emb",
        "paged_attention_fwd",
        "rms_norm",
        "multinomial_sampling",
        "silu_and_mul",
    ]

    def try_patch(op: str):
        def patch_fill_kv_cache():
            lmdeploy_kernels.fill_kv_cache = fill_kv_cache

        def patch_apply_rotary_pos_emb():
            lmdeploy_kernels.apply_rotary_pos_emb = apply_rotary_pos_emb

        def patch_paged_attention_fwd():
            lmdeploy_kernels.paged_attention_fwd = paged_attention_fwd

        def patch_rms_norm():
            lmdeploy_kernels.rms_norm = rms_norm

        def patch_multinomial_sampling():
            lmdeploy_kernels.multinomial_sampling = multinomial_sampling

        def patch_silu_and_mul():
            import lmdeploy.pytorch.kernels.cuda.activation as activation

            activation.silu_and_mul = silu_and_mul

        try:
            locals()[f"patch_{op}"]()
            print(f"patched dlblas implementation of {op}\n", end="")
        except KeyError:
            print(
                f"unknow op: {op}, supported ops: {DEFAULT_PATCH_LIST}\n",
                end="",
            )
        except AttributeError:
            print(f"op {op} is not implemented in dlblas\n", end="")

    for op in DEFAULT_PATCH_LIST:
        try_patch(op)


_patch_lmdeploy()

from lmdeploy import PytorchEngineConfig, pipeline
from lmdeploy.messages import GenerationConfig


def run_pipeline_chat_test(
    device_type,
):
    tp = 2
    hf_path = "/nvme/share_data/models/internlm2-chat-7b/"
    backend_config = PytorchEngineConfig(tp=tp, device_type=device_type)
    print("backend_config: ", backend_config)
    pipe = pipeline(hf_path, backend_config=backend_config)
    gen_config = GenerationConfig(do_sample=True, top_k=2)
    response = pipe(
        [
            "给我一首中文诗，需要添加标点符号，请用中文回答Give me a Chinese poem in Chinese"
        ],
        gen_config=gen_config,
    )[0].text
    print(response)
    assert "诗" in response and "《" in response and "》" in response
    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run_pipeline_chat_test("cuda")
    print("sucessfully!")
