# Triton
triton for dsa

## compile llvm project
```
git clone https://github.com/llvm/llvm-project.git
// triton下的llvm-hash.txt commit id
git reset --hard ed4e505c219fe6c7464ea5a056e90d8cd94c7332

cmake -G Ninja ../llvm  -DLLVM_ENABLE_PROJECTS="llvm;mlir"    -DLLVM_BUILD_EXAMPLES=ON    -DLLVM_TARGETS_TO_BUILD="X86X86;NVPTX;AMDGPU"     -DCMAKE_BUILD_TYPE=Release  -DLLVM_ENABLE_ASSERTIONS=ON       -DLLVM_INSTALL_UTILS=ON

ninja -j64
```


## 编译 triton && triton
```
export LLVM_BUILD_DIR={path-of-llvm-project}/build
bash compile.sh
export PYTHONPATH=$PWD/third_party/triton/python
export PATH=$PWD/third_party/triton/build/third_party/triton_shared/tools/triton-shared-opt/:$PATH
```


## 测试
```
cd python/op
python softmax.py
```

## 刷新code格式
```
bash format.sh
```

# 华为昇腾芯片
## 编译
```
# set LLVM_INSTALL_PREFIX
bash compile_on_ascend.sh
```
## 设置昇腾后端
```
import triton.backends.dicp_triton.driver as dicp
triton.runtime.driver.set_active(dicp.DICPDriver('ascend'))
```
## 测试
```
cd python/op
python softmax.py
```

# 寒武纪芯片
## 编译
```
bash compile_on_mlu.sh
```
## 设置寒武纪后端
```
import triton.backends.dicp_triton.driver as dicp
triton.runtime.driver.set_active(dicp.DICPDriver('mlu'))
```
## 测试
```
cd build/triton/tutorials
python 01-vector-add.py
```

# 支持模型框架列表

## LMDeploy
### 寒武纪云端智能加速卡
| 模型              | 类型 | FP16/BF16 | KV INT8 | KV INT4 | W8A8 | W4A16 |
| ---               | ---  | ---       |    ---  | ---     | ---  | ---   |
| internlm2-chat-7b | LLM  | YES       |         |         |      |       |
| internlm2_5-7b    | LLM  | YES       |         |         |      |       |
| Qwen2-7B          | LLM  | YES       |         |         |      |       |
| Llama-2-7b-hf     | LLM  | YES       |         |         |      |       |
