# 面向 Triton 用户的 MACA Gluon 与 Layout Autotune

## 1. 为什么需要 Gluon，为什么还需要 layout autotune

### Triton 的边界：算法可见，物理流水大多不可见

Triton 让用户描述逻辑 tile、global pointer 与 launch 配置，编译器决定线程、warp、寄存器、
shared memory 和 layout 的大部分细节。FlashAttention、GEMM pipeline 等 kernel 还需要控制：

- 一块数据是 `Global -> register -> shared -> register`，还是另一路径？
- K/V tile 计算时，下一块 K 何时预取？
- shared memory 何时存 Q、K/V 或 O，何时才能覆盖？
- dot 输入、累加器和 global store 的 lane/warp ownership。

普通 Triton 尽量把这些细节隐藏，因此用户可以调整 `BLOCK_*`、`num_warps`、`num_stages`，
却不能稳定、细粒度地规定完整流水和 shared 的生命周期。Gluon 的目的就是把这些控制点提升到
DSL：用户显式写 storage、view、数据移动、同步和 dot 数据依赖；编译器仍负责把程序落到具体
硬件指令。

### 新挑战：layout 既抽象又容易组合爆炸

有了这些控制能力，另一个问题随即出现：layout 不能简单理解为行主序或列主序。它描述的是一个
逻辑 tile 的元素如何分布到 CTA 的 warp/lane、寄存器和 shared 地址中。一个 layout 是否可用，
同时取决于：

```text
dot 与 global load/store 约束
        + shared swizzle、view 与 producer→consumer 地址映射
        + C500 指令与资源限制
```

手写 layout 会将算法与硬件路径耦合；LLM agent 难以从局部语法可靠推导 lane/warp ownership；直接枚举
`warp`、`order`、`vec`、`phase` 和 shared view 的组合又会快速膨胀。

所以本项目引入 **compiler-managed layout + layout autotune**。它不是让 Python 或 agent 猜一堆
layout，而是让编译器根据完整数据流构造有限、已验证的候选；Python runtime 只编译和测量这些
候选，选择胜者。

## 2. 编译与使用

### 2.1 获取构建所需的 MetaX LLVM

mcTriton 的构建需要 MetaX LLVM。该 LLVM 可以从沐曦开发者社区提供的 mcPytorch 安装包中取得：

1. 访问 [沐曦开发者社区的 torch 包搜索页](https://developer.metax-tech.com/softnova/search?package_name=torch)，搜索 mcPytorch 安装包。
2. 选择 mcPytorch 2.4、2.6 或 2.8 中与当前 Python/MACA 环境匹配的安装包。例如可下载
   [maca-pytorch2.8-py312-3.7.2.0-x86_64.tar.xz](https://developer.metax-tech.com/softnova/search?package_name=maca-pytorch2.8-py312-3.7.2.0-x86_64.tar.xz)。
3. 解压 mcPytorch 安装包，找到其中的 `metax_llvm*.tar.xz`；再次解压它，得到 `metax_llvm` 目录。

将该目录放到可访问的位置，例如本文环境使用：

```text
/datapool/kezengxiang/2026/third_party/metax_llvm_3.8.0/
```

### 2.2 构建 mcTriton

先确保 `MACA_PATH` 已指向可用的 MACA 安装目录，然后在 mcTriton 根目录执行：

```bash
cd ${$mcTriton}

bash maca/maca_tools/build_triton.sh \
    --llvm /datapool/kezengxiang/2026/third_party/metax_llvm_3.8.0/ \
    -m "${MACA_PATH}"
```

整体构建流程与 [MetaX-MACA/mcTriton](https://github.com/MetaX-MACA/mcTriton) 相同；这里新增的
Gluon DSL 和 C500 layout autotune 路径会随同 `libtriton` 一起构建和安装。

### 2.3 运行两个示例

确认当前 Python 导入的是刚构建安装的 Triton 后，可运行：

```bash
python -c 'from triton.experimental import gluon; print(gluon.__file__)'

# 显式 Global→shared copy、四槽流水的 TN matmul
python python/tutorials/gluon/metax/04-matmul-tn-1.py

# FlashAttention forward：主例子
python python/tutorials/gluon/metax/07-flash-attention-forward.py \
    --seq-len 128 1024 20480 --dtype fp16
```

第一个示例默认运行正确性用例；添加 `--benchmark` 会执行它定义的 benchmark 形状。第二个示例会验证
O 与 LSE；首次运行还可能包含 JIT 和 layout autotune 的准备时间，因此应将首次耗时与稳定运行时间
分开观察。

## 3. 整体方案：写流水语义，让系统处理 layout 选择

```text
Triton 算法 / PyTorch launcher
             │  同样使用 kernel[grid](...)
             ▼
Gluon DSL
  写清楚 tile、Global↔register↔shared、view、barrier、dot
             │
             ▼
compiler-managed layout
  为这些语义路径补全并筛选可行的物理 layout
             │
             ▼
layout autotune runtime
  测量可行方案，缓存本 workload 的胜者
             │
             ▼
MACA kernel
```

读者只需先记住边界：**你写“数据怎样流、何时复用”；系统决定“每个值由哪些 lane/warp 以何种
layout 携带”，并用实测选择可行方案。** `@gluon.jit` 的 `layout_autotune` 当前默认开启；它的意义
是选择已有语义程序的 layout 实现，不改变 Python 里写出的算法、同步和内存生命周期。

## 4. 与 Triton 的区别，以及 Gluon DSL 的层次

| 维度 | Triton 的常见写法 | 此处 Gluon 的写法 |
|---|---|---|
| 启动方式 | `kernel[grid](...)` | 相同 |
| kernel decorator | `@triton.jit` | `@gluon.jit` |
| 基础算子 | `tl.arange/load/store/where` | `gl.arange/load/store/where`，语义接近 |
| shared memory | 大多由编译器隐式处理 | `gl.local_alloc()` 显式申请 shared arena |
| shared 访问 | 很少直接操作 shared descriptor | `smem.store(x)`、`smem.load()` |
| 同一 storage 的不同逻辑形状 | 通常不暴露 | `index`、`reshape`、`_reinterpret` |
| CTA 内同步 | 多数被抽象 | `gl.metax.barrier_shared()` 等显式描述 |
| Global→shared pipeline | 后端尽量决定 | 可用 C500 `async_copy_global_to_shared` 显式提出 |
| layout 选择 | 通常完全隐式 | 可写显式 layout；本项目更常使用 compiler-managed layout/autotune |

通用 `gl.*` 用于 tile 计算和 compiler-managed shared allocation；descriptor API 用于 shared view；
`gl.metax.*` 只保留 MACA/C500 的硬件语义。`gvm_arrive/iglp/sched_bound` 虽然直接位于该
namespace，仍只应在需要复刻显式 C500 调度的 kernel 中使用。初学时先写对数据依赖和 shared
生命周期，再加入后端特化 pipeline 指令。

## 5. 主要 DSL：先学“怎样描述流水”

| DSL | 表达的语义 | Flash `kernel2` | TN matmul 辅助例子 |
|---|---|---|---|
| `gl.load` / `gl.store` | global memory 与寄存器之间的读写 | 读 Q/K/V，写 O/LSE | 读写 A/B/C pointer tile |
| `gl.local_alloc` | 分配一块可分槽的 shared storage | 两个 `[64,128]` buffer 的 arena | A、B 各四个 pipeline slot |
| `descriptor.index(i)` | 选择第 `i` 个 shared slot/view | 选择 K 或 V 所在 buffer | 选择 0~3 号 A/B staging slot |
| `descriptor.store/load` | register ↔ shared 的显式移动 | Q/K/V/O staging | 从已到达的 A/B shared slot 取 MMA operand |
| `_reinterpret` | 不搬数据，重描述同一 shared storage 的 dtype/shape/view | arena 被看成 Q、K 转置、V fragment、O | 本例不依赖它，突出直接的 fixed slot pipeline |
| `gl.dot` | 矩阵乘的语义与数据依赖 | QK、PV 两类 dot | A/B 32×32 子块累加到 C 子块 |
| `gl.slice(source, shape, offsets)` | 从 register tensor 或 pointer/mask tile 取静态子 tile | 将 `[64,D]` K/V 切为两个 `[32,D]` N32 片段 | 将 A/B pointer、mask 与 accumulator 切成 32 行/列子块 |
| `gl.slice_update(base, update, offsets)` | 将静态子 tile 写回完整 distributed tensor，并返回新 tensor | 本例不需要回写大 accumulator | 将 16 个更新后的 32×32 C 子块写回 128×128 `loop_acc` |
| `barrier_shared` | CTA 内 shared 的发布、读取或复用边界 | Q→K、K/V→O 的 epoch 边界 | 每批 async copy 与 subsequent shared load 之间 |
| `async_copy_global_to_shared` | C500 的 Global→shared 异步 copy 意图 | 本例未直接调用 | A/B 的四槽 copy pipeline |
| `gl.metax.barrier/barrier_shared` | C500 正式的可见性、复用边界 | kernel2 主要用 `barrier_shared` | async copy 和 shared consumer 之间的同步 |
| `gl.metax.gvm_arrive/iglp/sched_bound` | 复刻手写 C 调度的实验入口 | 自动 kernel2 不使用 | 显式基线保留既有到达计数和调度提示 |

`pipeline="cpasync"` 是 launch 配置，不是源码已经发出某一种具体异步 copy 或 wait 指令的证明。要确认
最终是否 lower 为期望的 C500 指令，仍应检查 TTGIR/LLIR/最终二进制。

### 5.1 `@gluon.jit`、`gl.constexpr` 与 launcher：与 Triton 相同的启动接口

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def kernel(x_ptr, BLOCK: gl.constexpr):
    ...

kernel[grid](x, BLOCK=128, num_warps=4)
```

PyTorch tensor 仍作为 global pointer 传入，`grid` 仍决定 CTA 数，`gl.constexpr` 仍表示编译期常量。
Gluon 的新增内容发生在 kernel 内部：当 tile 经过 shared 或 dot 时，代码还要描述 storage、view 和
同步关系。

### 5.2 `program_id`、`arange`、pointer：先确定一个 CTA 的工作范围

Flash kernel2 的 `pid_m` 和 `pid_bh` 分别选择 Query tile 与 batch/head；TN matmul 的一维 `pid`
再映射为 `(pid_m, pid_n)`。随后都用 `gl.arange` 生成 tile 内坐标，用 stride 组成 pointer tile：

```python
offs_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N)
ptrs = base + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
```

`gl.load(ptrs, mask=..., other=0.0)` 得到寄存器 tensor，`gl.store(ptrs, value, mask=...)` 写回 global。
这部分可按普通 Triton 理解；后续 DSL 的重点是如何让这个寄存器 tensor 有计划地进入 shared、dot 或
下一轮流水。

### 5.3 `local_alloc` 与 descriptor：shared 不是普通 tensor

`gl.local_alloc(dtype, shape, num_buffers=N)` 申请 shared storage，返回的是 **shared descriptor**，
不是已经装着寄存器值的 tensor。descriptor 的操作正是在描述数据流：

```python
smem = gl.local_alloc(x_ptr.dtype.element_ty, [M, K], num_buffers=2)
smem.index(0).store(x_register)      # register -> shared slot 0
x_again = smem.index(0).load(...)    # shared slot 0 -> register
```

`num_buffers` 是流水槽数量。kernel2 用 2 个 slot 保存不同 epoch 的 Q/K/V/O；TN matmul 用 4 个
slot，使 copy 与 compute 交错。每个 slot 都应明确：写入者、读取者以及允许覆盖它的同步点。

### 5.4 `index`、`reshape`、`_reinterpret`：改变 view，不搬数据

- `index(i)` 选择某一个 pipeline slot；
- `reshape(shape)` 改变同一 descriptor 的逻辑 shape；
- `_reinterpret(dtype, shape)` 将同一物理 shared storage 作为另一个 dtype/shape/view 使用。

这些操作不会生成 global copy。kernel2 将同一 `shared_arena` 分别看作 Q/O 的 `[M,D]` view、K 的
`[D,N]` view 与 V 的 `[2,32,D]` view；这是“同一块地址在不同 epoch 服务不同 consumer”的直接表达。
它也意味着 view 之间不是任意可换：要保证存储未被仍在运行的读者使用，并让 compiler-managed layout
处理 view 与 consumer 的物理兼容性。

### 5.5 `gl.slice`：静态地取出一个 register 子 tile

`gl.slice(source, shape, offsets)` 从 distributed tensor 中取出静态子块。`shape` 是结果 shape，
`offsets` 是每一维起点；二者都必须是编译期已知的整数列表。它不从 global memory 重新读取数据，
也不是 Python 的动态切片，而是在 Gluon IR 中建立一个 register 子 tile/value。shared descriptor
不走这个 op：选择 pipeline slot 用 `desc.index(...)`，同 rank 子区间用
`desc.slice(start, length, dim)`。

Flash kernel2 中，当前 K register tile 的 shape 是 `[64, D]`。M64 路径把它切成两个 `[32, D]`：

```python
k0 = gl.slice(k_register, [32, BLOCK_D], [0, 0])
k1 = gl.slice(k_register, [32, BLOCK_D], [32, 0])
```

随后 `k0/k1` 各自 `permute(1, 0)` 并写入两个 K shared fragment。V 也以完全相同的 offsets
切为两块 `[32, D]`，但不需要 `permute`。这使每个 N32 QK/PV dot 都有明确的输入范围。

TN matmul 中，`slice` 还有两种用法：

```python
# 从 [BLOCK_M, BLOCK_K] 的 pointer/mask tile 取一段 [32, BLOCK_K]
a_ptrs0 = gl.slice(a_ptrs, [32, BLOCK_K], [0, 0])
a_mask0 = gl.slice(a_load_mask, [32, BLOCK_K], [0, 0])

# 从 [128, 128] accumulator 取左上角的 [32, 32]
c00 = gl.slice(loop_acc, [32, 32], [0, 0])
```

前一组切片使 pointer、mask 与 `a_smem.index(0)` 的 `[32, BLOCK_K]` 形状一致，能够发起这一小块
async copy；后一组切片让一个大 accumulator 的每个 32×32 子块分别传给 `gl.dot`。

### 5.6 `gl.slice_update`：把子 tile 写回一个大 tile

`gl.slice_update(base, update, offsets)` 的输入是完整 tensor `base`、与目标区域同 shape 的
`update` 和静态 `offsets`。它返回一个新的完整 tensor，其中指定区域被 `update` 替换；因此调用处
需要重新赋值：

```python
c00 = gl.dot(loop_a0, loop_b0, c00, input_precision="tf32")
loop_acc = gl.slice_update(loop_acc, c00, [0, 0])
```

TN matmul 对 16 个 32×32 C 子块重复这个过程，最终恢复为完整的 128×128 `loop_acc`。`slice` 与
`slice_update` 成对使用时，offset 和 shape 必须对应；这也是理解该示例里大量 `c00` 到 `c33` 变量的
最直接方式。

### 5.7 `gl.dot`：表达子块矩阵乘和累加依赖

`gl.dot(a, b, acc)` 表示 `acc += a × b`。先检查三个 tensor 的 shape 与数据来源：Flash 的两次 QK
和两次 PV dot 使用 N32 fragment；TN matmul 的每次 dot 使用一个 A 子块、一个 B 子块和一个 C 子块。
寄存器 fragment 的 lane/warp 物理分配不在 DSL 中手写，由 layout 机制根据完整路径处理。

### 5.8 `barrier_shared`：声明 shared 的三个时刻

对每个 shared slot，都应能指出三个时刻：

```text
writer 完成 store/copy  →  barrier_shared  →  reader load/use 完成
                                              ↓
                                  允许下一 epoch 覆盖或 reinterpret
```

Flash kernel2 中，Q 写入 sQ 后同步才能被所有线程读；读完后再次同步，slot 0 才能改作 sK。循环中，
sK 发布后才能做 QK，sV 发布后才能做 PV；结束后 K/V arena 才能改为 sO。`barrier_shared` 不是性能
装饰，而是 shared 可见性与复用安全性的程序语义。

### 5.9 C500 pipeline DSL：先从 matmul 的显式 copy 看起

TN 手写基线的 `gl.metax.async_copy_global_to_shared(smem, ptrs, ...)` 直接表达固定调度的
Global→shared 异步 copy；随后用 `gl.metax.gvm_arrive` 与正式的
`barrier`/`barrier_shared` 划出 copy producer 和 shared consumer 的边界。它还使用
`gl.metax.iglp` 与 `gl.metax.sched_bound` 描述 C500 特有的调度提示。

这些显式调度接口不进入 compiler-managed 主路径，也不是 layout 候选输入。推荐学习顺序是：先用
通用 `local_alloc`、普通 `store/load` 和正式 `barrier_shared` 写对 storage 生命周期；只有复刻已有
手写 C pipeline 时，才显式加入整条 producer→同步→consumer 调度。

Gluon 的 shared 读取统一写成 `descriptor.load()`，不暴露 `dtype`、`intrinsic`、
`is_constant_offs` 或 `mma_mode`。`bsm_perm()` 是 C500 特有的“整数 carrier 重排并 cast”
语义；当它直接消费一个逻辑 shared load 时，编译器在 layout 具体化后自动把该 load 物化为
同 shape/layout 的 i32 carrier。Python 不新增 `bsm_load`，也不把 `mmaMode=2` 绑进
`bsm_perm` 合同。物化由独立的 C500 BSM legalization pass 完成；当前后端所需的
split-LDS mode 只由该 pass 在已验证的 `local_load -> bsm_perm` 物理路径上内部设置，
不会成为 DSL 参数或 layout 搜索维度。

## 6. 主例：FlashAttention kernel2 如何使用这些 DSL

### 6.1 一个 CTA 做什么

`flash_fwd_hdim128_generic_kernel` 以二维 grid 运行：`pid_m` 选择 Query tile，`pid_bh` 选择
batch/head。一个 CTA 负责一个 `(batch, head, BLOCK_M)` 的 Q 行块，依次扫描 K/V 的 N64 tile。
它将 `QK^T`、online softmax、`PV`、O/LSE 写回融合在单 kernel 中，不把完整 score 或 probability
矩阵写回 global memory。

```text
Q: Global → rQ → sQ → rQ_for_score ────────────────┐
                                                     │
每轮 N64：                                            ▼
  rK(current) → sK^T → barrier → rK operand → QK dot → score
  Global V → rV → sV fragments → barrier → rV operand ┘
                                                     │
  score → online softmax(m_i, l_i) → P → PV dot → acc_o
  同时 Global next-K → rK(next)，供下一轮使用

结束：acc_o / l_i → sO → barrier → coalesced rO → Global O
      m_i + log(l_i)                              → Global LSE
```

### 6.2 `shared_arena`：先规划生命周期，再看 layout

主 kernel 在一开始申请：

```python
shared_arena = gl.local_alloc(
    q_ptr.dtype.element_ty, [BLOCK_N, BLOCK_D], num_buffers=2
)
```

对于 fp16/bf16、`BLOCK_N=64`、`D=128`，它是两个 `[64,128]` slot，总计 32 KiB。关键并非大小，
而是 epoch 复用：

```text
prologue:    slot 0 = sQ                 slot 1 = unused
main loop:   slot 0 = sK                 slot 1 = sV
epilogue:    slot 0 = sO                 slot 1 = unused
```

这就是 Gluon 对流水的细粒度控制：同一物理 storage 不同时刻具有不同逻辑角色，但每次角色切换都必须
有清晰的读写完成与同步边界。

### 6.3 `_reinterpret`：同一地址的多个合法逻辑 view

kernel2 中，slot 0/1 分别被重新解释为：

```python
q_smem      # [BLOCK_M, BLOCK_D] 的 Q view
k_tn_smem   # [BLOCK_D, BLOCK_N] 的 K 转置 view
v_fragments # [2, 32, BLOCK_D] 的 V 分片 view
o_smem      # [BLOCK_M, BLOCK_D] 的 O view
```

`_reinterpret` 不产生 copy，也不执行真实 transpose；它只重述同一 shared storage 的逻辑 shape/view。
例如 K 被写进 `k_tn_smem`，之后 QK dot 读到的就是适合 `Q × K^T` 的 shared view。是否能把这些
view 对应到正确物理地址、是否需要 layout transfer，是 compiler-managed layout candidate 必须证明的
事情，而不是让用户在 Python 中猜 lane mapping。

### 6.4 一次 K/V tile 的工作

辅助函数 `_flash_fwd_tile` 做一轮 N64：

1. 当前 `k_register` 写入 `k_tn_smem`；
2. 从 global 读当前 V 到 `v_register`；
3. `barrier_shared()` 后从 sK 读两个 N32 fragment，做两次 QK dot；
4. V 写到两个 shared fragment；
5. 预取 next K 到寄存器；
6. 再经过 `barrier_shared()`，计算 online softmax；
7. 从 sV 读两个 fragment，做两次 PV dot，更新 `acc_o`；
8. 返回 `next_k_register`，让下一轮少一次关键路径上的 global K load。

两次 N32 dot 不是语义上的重复：它让 QK 输出和 PV 输入沿着当前 C500 可 lower 的 fragment 路径闭合。

### 6.5 为什么 O 还要经过 shared

循环结束后，`acc_o` 先写入 `o_smem`，同步后再读成 `out_coalesced`，最后写 global O。这个
shared round-trip 的目的不是数值计算，而是把 PV accumulator 的寄存器 ownership 转换为适合
D-contiguous global store 的 writer ownership。它展示了 Gluon 的一个核心观点：layout 不只是 dot
之前的细节，最终 global store 也是一个物理边界。

## 7. 辅例：TN matmul 如何把流水控制写得更显式

`04-matmul-tn-1.py` 计算 `C = A × B`。其中 B 的逻辑 shape 为 `(K,N)`，但测试构造中用
`(N,K).transpose(0,1)` 获得 K 连续访问。它比 Flash 核更直接地展示了 C500 pipeline DSL。

### 7.1 四槽 A/B shared pipeline

```python
a_smem = gl.local_alloc(..., [32, BLOCK_K], num_buffers=4)
b_smem = gl.local_alloc(..., [BLOCK_K, 32], num_buffers=4)
```

128×128 的输出 tile 被划成 16 个 32×32 C 子块；A/B 的当前 K tile 也被拆成多个 32 行/列片段。
程序把 A 和 B 分别 staging 到四个 slot，并在计算当前子块时覆盖已经安全释放的旧 slot：

```text
Global A/B
  │ async_copy_global_to_shared(slot i)
  ▼
A/B shared slot 0..3
  │ gl.metax.gvm_arrive + barrier / barrier_shared
  ▼
register A/B fragment
  │ gl.dot
  ▼
32×32 C fragments
  │ slice_update
  ▼
128×128 accumulator → Global C
```

这份 matmul 显式调用 `gl.metax.async_copy_global_to_shared`，而不是只写普通 `gl.load` 后再
`smem.store`。它还出现 `gl.metax.gvm_arrive`、正式的 `barrier/barrier_shared`，以及
`gl.metax.iglp/sched_bound`：这些接口表达 copy 到达、同步或调度边界。它们不能被当成随意可调的数字；
应先保持示例已有的 producer→同步→consumer 关系，再基于 IR/性能证据修改。

### 7.2 它为何只是辅助例子

matmul 展示显式 four-slot copy/compute pipeline；Flash kernel2 展示复杂算法中 shared storage 的
生命周期、view 与同步。两者都由 DSL 描述数据流；layout 的物理细节由系统处理，runtime 选择更快的
可行方案。

## 8. 实际使用建议

1. 保持 Triton 的 grid、block 参数和数值验证逻辑，先确定一个 CTA 的 tile。
2. 用 `local_alloc`、view、`store/load` 表达 staging；覆盖 shared 前确认旧读者已越过 barrier。
3. 先用 `gl.dot` 表达数学依赖，保留默认 `layout_autotune`；将首次运行时间与稳态时间分开看。
4. 先验证 O/LSE 或 C，再检查 TTGIR/LLIR 的 layout、同步、资源和最终指令。

## 小结

Gluon 解决的是 Triton 难以细粒度表达流水的问题：开发者可以明确写出 Global、register、shared、
dot 与 barrier 的关系。layout autotune 解决的是另一个更底层的问题：layout 过于抽象、难以人工或
LLM agent 可靠推导，且直接枚举组合会爆炸。因此本项目不要求用户手写所有 layout，而是让编译器
从完整数据流中生成并验证封闭候选，让 runtime 以实测选择最佳 variant。

读这两个示例时，可始终用一句话检查理解是否正确：**Gluon 源码负责说明数据如何流动和何时可复用；
compiler 负责证明它以什么物理 layout 流动；runtime 负责决定哪个已证明的完整实现最快。**
