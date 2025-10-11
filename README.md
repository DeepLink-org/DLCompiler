# 介绍
DLCompiler是上海人工智能实验室（上海 AI 实验室）DeepLink 团队开源扩展 Triton 的深度学习编译器：
- 跨架构 DSL 扩展：通过扩展 DSL，让 DSA 芯片（昇腾芯片）也能享受 GPU 级的编程体验和性能，成为 “跨架构 AI Kernel DSL” 。
- 智能自动优化：实现智能核间调度，充分释放多核算力；结合创新的访存合并优化，将离散访问自动重组为高速连续访问，大幅提升算子性能与带宽利用率。
<img width="1586" height="992" alt="身位图-昇腾" src="https://github.com/user-attachments/assets/59c195cc-2702-4d5a-8559-3bed1722281e" />


# 编译使用
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

# 昇腾芯片
## 环境准备
准备昇腾设备上环境，可以参考昇腾的链接：https://gitee.com/ascend/triton-ascend
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
bash compile_shared.sh apply_patch=true     # 如果不应用patch，可以直接执行 bash compile_shared.sh，如果想要尝试使用新版triton_shared，编译时加上compile_triton_shared=true
# TODO：下面的内容尽可能做到脚本中，不要单独设置，太复杂。
export TRITON_SHARED_OPT_PATH=$PWD/third_party/triton/python/build/cmake.linux-aarch64-cpython-3.10/third_party/dicp_triton/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt
export DICP_OPT_PATH=$PWD/third_party/triton/python/build/cmake.linux-aarch64-cpython-3.10/third_party/dicp_triton/tools/dicp_triton_opt/dicp_opt
export LOWER_BY_TTSHARED=1
# 如果更新了最新版本的triton-shared-opt，需要修改TRITON_SHARED_OPT_PATH
# export TRITON_SHARED_OPT_PATH=$PWD/third_party/build/triton/build/cmake.linux-aarch64-cpython-3.10/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt
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
