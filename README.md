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
## 环境准备
准备华为设备上环境，可以参考华为的链接：https://gitee.com/ascend/triton-ascend
### 安装ascend cann
1. 要求CANN 版本 > 8.2.RC1.alpha002
2. 社区下载链接：https://www.hiascend.com/developer/download/community/result?module=cann
3. 社区安装指引链接：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit

### 安装依赖
```
pip install attrs==24.2.0 numpy==1.26.4 scipy==1.13.1 decorator==5.1.1 psutil==6.0.0 pytest==8.3.2 pytest-xdist==3.6.1 pyyaml pybind11
```
### 安装torch_npu
```
pip install torch_npu==2.6.0rc1
```
## 编译
```
# set LLVM_INSTALL_PREFIX
bash compile_on_ascend.sh
```

### ttshared pipeline
```
bash compile_shared.sh
export TRITON_SHARED_OPT_PATH=$PWD/third_party/triton/python/build/cmake.linux-aarch64-cpython-3.10/third_party/dicp_triton/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt
export DICP_OPT_PATH=$PWD/third_party/triton/python/build/cmake.linux-aarch64-cpython-3.10/third_party/dicp_triton/tools/dicp_triton_opt/dicp_opt
export LOWER_BY_TTSHARED=1
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
