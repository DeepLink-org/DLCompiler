## 源码编译

### 1. 依赖安装

##### 1.1 沐曦软件栈

参考[环境准备](https://github.com/MetaX-MACA/mcPytorch/blob/2.4/README.md#1-%E5%AE%89%E8%A3%85)准备沐曦软件栈环境。

### 2. 编译 mcTriton

##### 2.1 拉取代码

##### 2.2 编译

``` shell
bash ./maca/maca_tools/build_triton.sh --llvm /path-to-metax-llvm -m ${MACA_PATH}
```
