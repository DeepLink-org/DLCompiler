import os
import torch
from dlblas.kernels.fill_kv_cache import fill_kv_cache
from dlblas.kernels.apply_rotary_pos_emb import apply_rotary_pos_emb
from dlblas.kernels.paged_attention import paged_attention_fwd
from dlblas.kernels.rms_norm import rms_norm
from dlblas.kernels.multinomial_sampling import multinomial_sampling
from dlblas.kernels.activation import silu_and_mul

from lmdeploy import PytorchEngineConfig, pipeline
from lmdeploy.messages import GenerationConfig
import json
import yaml

import argparse
import traceback


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
            print(f"Patched dlblas implementation of {op}\n", end="")
        except KeyError:
            print(f"Unknown op: {op}, supported ops: {DEFAULT_PATCH_LIST}\n", end="")
        except AttributeError:
            print(f"Op {op} is not implemented in dlblas\n", end="")

    for op in DEFAULT_PATCH_LIST:
        try_patch(op)


_patch_lmdeploy()


def run_pipeline_chat_test(model_name, model_path, common_prefix, device_type="cuda"):
    tp = 1
    backend_config = PytorchEngineConfig(
        tp=tp, device_type=device_type, eager_mode=True, download_dir=common_prefix
    )
    if os.path.exists(model_path):
        pipe = pipeline(model_path, backend_config=backend_config)
    else:
        pipe = pipeline(model_name, backend_config=backend_config)
    gen_config = GenerationConfig(do_sample=False, top_k=1)
    
    response = pipe(
        ["Please introduce Shanghai."],
        do_preprocess=False,
        gen_config=gen_config,
    )[0].text
    print(f"Response from model at {model_path}: {response}")

    return response

def run_pipeline_image_test(model_name, model_path, common_prefix, device_type="cuda"):
    tp = 1
    backend_config = PytorchEngineConfig(
        tp=tp, device_type=device_type, eager_mode=True, download_dir=common_prefix
    )
    if os.path.exists(model_path):
        pipe = pipeline(model_path, backend_config=backend_config)
    else:
        pipe = pipeline(model_name, backend_config=backend_config)
    
    from lmdeploy.vl import load_image
    image_path = '/home/pujiang/zhanglantian/Triton/python/dlBLAS/tests/e2e/test_lmdeploy/tiger.jpeg'
    image = load_image(image_path)
    response = pipe(("describe this image", image)).text
    print(f"Response from model at {model_path}: {response}")

    return response

def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint_file, tested_models):
    with open(checkpoint_file, "w") as f:
        json.dump(tested_models, f, indent=4)


def test_model(model_name, model_path, common_prefix, device_type="cuda", checkpoint_file="checkpoint.json"):
    tested_result = load_checkpoint(checkpoint_file)
    
    if model_name in tested_result and tested_result[model_name]["status"] == "completed":
        print(f"---------Skipping already tested model: {model_name}")
        return
    print(f"---------Testing model {model_name} at path: {model_path}")
    try:
        if "vl" not in model_name.lower():
            response = run_pipeline_chat_test(model_name, model_path, common_prefix, device_type)
        else:
            response = run_pipeline_image_test(model_name, model_path, common_prefix, device_type) 
        tested_result[model_name] = {"model_path": model_path, "status": "completed", "response": response}
    except Exception as e:
        print(f"---------Error testing model {model_name} at {model_path}: {e}")
        traceback.print_exc()
        tested_result[model_name] = {"model_path": model_path, "status": "failed", "response": str(e)}
    save_checkpoint(checkpoint_file, tested_result)
    print(f"---------Model {model_name} at {model_path} tested over")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="Name of the model")
    parser.add_argument("model_path", type=str, help="Path to the model weight data")
    parser.add_argument("checkpoint_file", type=str, help="Path to the checkpoint file")
    parser.add_argument("common_prefix", type=str, help="Common prefix where store the model")

    args = parser.parse_args()



    torch.manual_seed(1024)
    torch.cuda.manual_seed(1024)
    torch.cuda.manual_seed_all(1024)


    test_model(args.model_name, args.model_path, args.common_prefix, checkpoint_file=args.checkpoint_file, device_type="cuda")