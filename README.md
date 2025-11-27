# 介绍

DLCompiler是上海人工智能实验室（上海 AI 实验室）DeepLink 团队开源扩展 Triton 的深度学习编译器：

- 跨架构 DSL 扩展：通过扩展 DSL，让 DSA 芯片（昇腾芯片）也能享受 GPU 级的编程体验和性能，成为 “跨架构 AI Kernel DSL” 。
- 智能自动优化：实现智能核间调度，充分释放多核算力；结合创新的访存合并优化，将离散访问自动重组为高速连续访问，大幅提升算子性能与带宽利用率。
  <img width="1586" height="992" alt="身位图-昇腾" src="https://github.com/user-attachments/assets/59c195cc-2702-4d5a-8559-3bed1722281e" />

# 基于昇腾芯片

## pip安装

```bash
# 因包过大，超过pypi限制（我们也在申请更大的容量），暂时支持从github安装
pip install https://github.com/DeepLink-org/DLCompiler/releases/download/v0.0.2/dlcompiler-3.4.0-cp310-cp310-linux_aarch64.whl
# 也可以先下载，然后再安装
wget https://github.com/DeepLink-org/DLCompiler/releases/download/v0.0.2/dlcompiler-3.4.0-cp310-cp310-linux_aarch64.whl
pip install dlcompiler-3.4.0-cp310-cp310-linux_aarch64.whl
```

## 源码安装

### 安装ascend cann

1. 要求CANN 版本 >= 8.3.RC1
2. 社区下载链接：https://www.hiascend.com/developer/download/community/result?module=cann
3. 社区安装指引链接：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit

### 安装依赖

```bash
pip install attrs==24.2.0 numpy==1.26.4 scipy==1.13.1 decorator==5.1.1 psutil==6.0.0 pytest==8.3.2 pytest-xdist==3.6.1 pyyaml pybind11 nanobind
```

### 安装torch_npu

1. 要求torch_npu版本 >= 2.6.0

```bash
pip install torch_npu==2.6.0
```

### 编译命令

```bash
bash compile_shared.sh apply_patch=true     # 如果不应用patch，可以直接执行 bash compile_shared.sh
```

## 查看编译过程的mlir文件

```bash
export DLC_DUMP_IR=1 # 默认在当前目录下
```

## 测试

```bash
python ./test/ascend/passed_tests/test_silu_and_mul.py
```

# 基于寒武纪芯片

## 编译

```bash
bash compile_on_mlu.sh
```

## 测试

```bash
cd build/triton/tutorials
python 01-vector-add.py
```
