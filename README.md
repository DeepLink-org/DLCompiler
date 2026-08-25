# mcTriton

mcTriton 提供面向沐曦 GPU 的 Triton 编译链，并包含 MetaX C500 的 Gluon kernel 路径。

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

## Gluon C500：显式数据流与自动 layout

Gluon C500 支持两种 register-layout 写法：kernel 显式固定 layout，或者省略
register layout，由后端生成 candidates 并实测 winner。两种写法都使用 Python JIT 和
`kernel[grid](...)` 启动；global/register/shared 数据流、同步和 storage 生命周期仍由 kernel 显式定义。

### Gluon auto layout C500 性能快照

下表为 kernel-only 时间，`speedup = 对照 / Gluon`，大于 1 表示 Gluon 更快。
FA 为 2026-08-25 在当前 checkout 上的 C500 FP16 实测，shape 均为 `[1, 32, S, 128]`；
QKNorm+RoPE 是 2026-08-12 的 C500 BF16 历史快照。每行只在相同输入和该行标注的计时范围内比较。

| 算子 | 输入 | Gluon auto (ms) | 对照实现 (ms) | speedup | 正确性 |
| --- | --- | ---: | ---: | ---: | --- |
| FA forward | S=1K | 0.257344 | flash-attn 0.178423 | 0.693× | pass |
| FA forward | S=10K | 16.298496 | flash-attn 35.237888 | 2.162× | pass |
| FA forward | S=20K | 64.354561 | flash-attn 120.531197 | 1.873× | pass |
| FA forward | S=100K | 1575.032349 | flash-attn 2023.567871 | 1.285× | pass |
| FA backward main | S=1K | 0.790132 | flash-attn main 0.420250 | 0.532× | pass |
| FA backward main | S=10K | 61.699074 | flash-attn main 230.488602 | 3.736× | pass |
| FA backward main | S=20K | 246.381317 | flash-attn main 918.315648 | 3.727× | pass |
| FA backward main | S=100K | 6113.106934 | flash-attn main 10974.142379 | 1.795× | pass |
| QKNorm+RoPE | Qwen3-0.6B, `[40K, 4096]` | 0.6394 | vLLM-MetaX C 2.7026 | 4.227× | pass |
| QKNorm+RoPE | Qwen3-8B, `[40K, 6144]` | 1.0547 | vLLM-MetaX C 5.2710 | 4.997× | pass |
| QKNorm+RoPE | Qwen3-32B, `[40K, 10240]` | 1.9300 | vLLM-MetaX C 10.3935 | 5.385× | pass |
| QKNorm+RoPE | Qwen3-235B-A22B, `[40K, 9216]` | 1.9886 | vLLM-MetaX C 10.3634 | 5.211× | pass |

FA forward 使用 `triton.testing.do_bench` 的 mean，warmup=10、rep=30。FA backward 的
Gluon main 使用预计算 LSE+dPsum，flash-attn 数值是 backward 调用中 profiler 识别的主 kernel；
1K/10K/20K 使用 warmup=10、rep=30、profile-rep=10，100K 只使用 1/3/3 次做长序列校验。
QKNorm+RoPE 使用 p50，20 次 warmup 和 100 次 CUDA event 采样，原地输入恢复拷贝不计时。

FA forward/backward 的 1K/2K/4K/10K/20K/100K 六档均已通过正确性检查。
完整的参数、对照调用和计时代码在
[FA forward](python/tutorials/gluon/metax/10-flash-fwd-hdim128-generic-kernel-autotune.py) 和
[FA backward](python/tutorials/gluon/metax/12-flash-bwd-fused-kernel-autotune.py) 中。

### 1. 选择 layout 模式

| 模式 | kernel 写法 | decorator | 后端行为 |
| --- | --- | --- | --- |
| compiler-managed baseline | register 算子不传 `layout=` | `@gluon.jit` | 生成合法 baseline C0，不测量额外候选 |
| 自动 layout autotune | register 算子不传 `layout=` | `@gluon.jit(enable_gluon_layout_autotune=True)` | 生成 C0 和 layout candidates，cache miss 时实测 winner |
| 显式 register layout | 用 `gl.BlockedLayout`、`gl.SliceLayout`、`gl.MACAMmaLayout` 等 concrete encoding | `@gluon.jit` | module 走显式 layout lowering，不导出后置 candidates |

`enable_gluon_layout_autotune` 默认是 `False`。自动 layout 需要同时满足：

1. 在 `@gluon.jit(...)` 上显式设置 `enable_gluon_layout_autotune=True`；
2. module 中没有显式 register encoding。`gl.arange`、`gl.full`、`gl.zeros` 和 shared load
   的 register 结果都应省略 `layout=`；不要用 `gl.AutoLayout()` 代替省略参数。

后端按整个 module 判定路径。任一 operand、result 或 block argument 的 register tensor
带 concrete encoding，整个 module 就进入显式路径；即使同时打开 autotune，也不导出候选。
Shared memdesc 不属于 raw register tensor。

### 2. 运行显式/自动对照

仓库内的 Q/K RMSNorm + NeoX RoPE 提供同一算子的两个版本：

- [13：显式 layout](python/tutorials/gluon/metax/13-fused-qknorm-rope-neox-kernel.py)
  用 `BlockedLayout` 和 `SliceLayout` 固定 register 映射；
- [14：自动 layout](python/tutorials/gluon/metax/14-fused-qknorm-rope-neox-kernel-autotune.py)
  省略 register `layout=`，并显式打开 layout autotune。

自动版本只改两类代码：在 decorator 上打开 autotune，并从 register 算子上删除 `layout=`：

```diff
-@gluon.jit
+@gluon.jit(
+    do_not_specialize=["num_tokens"],
+    enable_gluon_layout_autotune=True,
+)
 def fused_qknorm_rope_neox_kernel(...):
-    head_layout = gl.BlockedLayout(...)
-    head_in_block = gl.arange(..., layout=gl.SliceLayout(1, head_layout))
-    dim = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, head_layout))
+    head_in_block = gl.arange(0, logical_warps_per_block)
+    dim = gl.arange(0, HEAD_DIM)
```

启动包装接口保持不变：

```python
fused_qknorm_rope_neox_packed(
    qkv, q_heads, kv_heads,
    q_weight, k_weight,
    cos_sin_cache, position_ids, eps,
)
```

`qkv` 是连续 BF16 `[T, (Q_HEADS + 2 * KV_HEADS) * 128]`，kernel 原地更新 Q/K，V 保持不变。
`q_weight` 和 `k_weight` 是 BF16 `[128]`，`cos_sin_cache` 是 `[max_position, 128]`，
`position_ids` 是 INT64 `[T]`。两个版本都使用 `num_warps=4`。

完整的定义、启动、PyTorch FP32 参考实现和正确性检查都在上面两个文件中。
下面的命令会运行显式版和自动版的 12 个 C500 正确性 case：

```shell
pytest -q \
  python/tutorials/gluon/metax/13-fused-qknorm-rope-neox-kernel.py \
  python/tutorials/gluon/metax/14-fused-qknorm-rope-neox-kernel-autotune.py
```

当前 checkout 的实测结果为 `12 passed`。

### 3. 显式 shared 数据流

自动 register layout 不会隐藏 shared pipeline。kernel 仍显式申请 storage、选择 buffer、
移动数据并放置同步点：

```python
smem = gl.local_alloc(
    x_ptr.dtype.element_ty,
    [BLOCK_M, BLOCK_K],
    num_buffers=2,
)
smem.index(0).store(x_tile)       # register -> shared buffer 0
gl.metax.barrier_shared()         # kernel 作者定义可见性和复用边界
x_again = smem.index(0).load()    # shared buffer 0 -> register
```

`gl.local_alloc` 返回的 descriptor shape 为 `[num_buffers] + logical_shape`；即使
`num_buffers=1` 也使用 `index(0)` 取 logical buffer。`reshape()`、`slice()` 和
`_reinterpret()` 只改变同一 storage 的 view，不复制数据，也不插入同步。

自动 layout 不会补 barrier、修复 async-copy mask、推导 buffer lifetime，也不会比较候选输出正确性。

### 4. 结构设计

#### 4.1 Raw register baseline 与候选导出

```text
@gluon.jit raw register IR
  -> Inliner -> StorageAliasLowering
  -> standard Triton-to-TritonGPU conversion -> Blocked baseline B0
  -> Coalesce -> F32DotTC -> RemoveLayoutConversions -> ThreadLocality
  -> AccelerateMatmul -> AlignMmaConsumers
  -> InsertRequireLayout -> PropagateLayout -> RewriteLocalAlias
  -> GluonToTritonGPU physical conversion
  -> existing C500 post-MMA TTGIR passes -> legal baseline C0

  if enable_gluon_layout_autotune=True:
    -> MMA Candidate     -> Expand(stage=0)
    -> Shared Candidate  -> Expand(stage=1)
    -> Blocked Candidate -> Expand(stage=2)
    -> verified standalone TTGIR leaves + manifest
```

`ExpandLayoutCandidates` 是同一个参数化 pass，在三个 stage 各调用一次。每个完整
assignment 都会 clone function、应用 layout replacement、重新推导相关 type 并 verify。
非法 leaf 不进入 manifest。baseline TTGIR 如果仍有 `tt.call`，导出器直接跳过候选导出；
没有候选时只使用 C0。

#### 4.2 显式 register-layout 路径

```text
@gluon.jit module containing a concrete register encoding
  -> Inliner -> StorageAliasLowering -> ResolveAutoEncodings
  -> RewriteLocalAlias -> GluonToTritonGPU physical conversion
  -> normal backend lowering
```

显式分支不运行 raw baseline 的
`AccelerateMatmul -> AlignMmaConsumers -> InsertRequireLayout -> PropagateLayout`，也不生成后置候选。

#### 4.3 核心文件

| 文件 | 职责 |
| --- | --- |
| `python/triton/experimental/gluon/_runtime.py` | `@gluon.jit` 开关、`do_not_specialize` dispatch 参数收集、launch 前 winner 选择 hook |
| `third_party/metax/backend/compiler.py` | 显式/raw TTGIR 路径选择、C500 baseline pass 顺序、候选导出接线 |
| `third_party/metax/backend/gluon_layout_autotune.py` | MMA/Shared/Blocked candidate 与参数化 Expand，standalone TTGIR 和 manifest |
| `python/triton/experimental/gluon/_layout_autotune.py` | workload key、候选编译/测量、进程内和磁盘 winner cache |
| `lib/Dialect/Gluon/metax/Transforms/` | baseline layout 约束、候选枚举、function clone/replace/verify 和 physical conversion |

### 5. 运行流程、winner cache 与副作用

C0 始终是 fallback。自动 layout 已导出非空 candidates 时，第一次非-warmup cache miss 的流程是：

1. 测量 C0；
2. 后台每批最多并行编译 4 个 standalone TTGIR candidate；
3. 编译批进入容量为 1 的队列，GPU 线程逐个串行测量；
4. 选择最快的可运行 candidate，把 digest 写入 `gluon-layout-winner.json`；
5. 用 winner 执行当前正式 launch。编译或测量失败的 candidate 会被跳过。

workload key 包含 JIT key、candidate domain、target、device 和所有 `do_not_specialize` 参数的
dispatch key。正整数 dispatch range 以 128 倍扩展，例如 `[1, 128)`、`[128, 16384)`。
pointer 对象地址和 concrete grid 不进入 key。

runtime benchmark 会在调用方传入的同一组参数上重复执行 C0 和 candidates，当前没有通用的
snapshot、reset、restore 或 correctness comparison。原地修改输入、累加输出或依赖输入初态的 kernel，
首次调优必须使用一次性存储：

```python
# 两组参数必须命中同一 JIT specialization、pointer alignment 和 dispatch key。
operator(*tune_args)   # disposable storage: 允许 C0/candidates 重复执行
operator(*real_args)   # cache hit: 只启动已选 winner
```

更多完整示例见 [Gluon C500 教程](python/tutorials/gluon/metax/00-introduction.md)、
[自动 FA forward](python/tutorials/gluon/metax/10-flash-fwd-hdim128-generic-kernel-autotune.py) 和
[自动 FA backward](python/tutorials/gluon/metax/12-flash-bwd-fused-kernel-autotune.py)。涉及 layout autotune 时，
以本 README 的显式 opt-in 合同为准。
