# Ascend NPU Auto-Tuning 迁移方案

> 迁移 triton-ascend auto_tuning 能力至 DLCompiler NPU 后端
>
> Date: 2026-05-27

---

## 1. 目标

将 triton-ascend 的 auto_tuning 能力完整迁移到 DLCompiler 的 NPU 后端，满足以下约束：

1. **backend=ascend 时**，`@triton.autotune` / `@triton.max_autotune` 走 triton-ascend 增强路径（AST 自动 tiling + 编译选项搜索 + 并行编译 + NPU benchmark）
2. **其他 backend 时**，保持社区版 triton autotune 行为不变
3. **不引入 triton-ascend 包依赖**：源码复用至 DLCompiler 内部
4. **最小化 triton 修改**：仅通过 `.patch` 在 `triton/__init__.py` 补充 `max_autotune` fallback

## 2. 设计模式：Dispatcher + Proxy Hook

### 2.1 为什么不用直接 Monkey-patch 硬编码？

直接硬编码的问题：
- `triton.autotune = ascend_autotune` 把 triton 模块引用和 ascend 实现强耦合
- 扩展性差：新增 backend 需要重写 hook 逻辑
- 可测试性差：无法在运行时切换策略

### 2.2 极简扁平模块（无类、无注册表）

> **为什么去掉 `AutotuneDispatcher` 类？**
> 运行时只会存在一种 backend（ascend 或 其他），不会同时存在多个。既然
> 没有多策略共存的需求，单例类、`_instance`、`_impls` dict 都是不必要的
> 包装。直接改为模块级变量 + 函数是最贴合实际约束的选择。
>
> 同时：proxy 必须在模块 import 时就完成替换，而不是依赖一个
> `_ensure_proxies_installed()` 函数在运行时再调用。

```python
# ascend_autotune_hooks.py
import triton

_USE_ASCEND = False          # 全局标志
_ASCEND_AUTOTUNE = None      # lazy-loaded
_ASCEND_MAX_AUTOTUNE = None  # lazy-loaded


def _ascend_autotune_fn():
    """首次 ascend 调用时懒加载。"""
    global _ASCEND_AUTOTUNE
    if _ASCEND_AUTOTUNE is None:
        from .ascend_autotune_runtime.autotuner import autotune as _fn
        _ASCEND_AUTOTUNE = _fn
    return _ASCEND_AUTOTUNE


def _ascend_max_autotune_fn():
    """首次 ascend 调用时懒加载。"""
    global _ASCEND_MAX_AUTOTUNE
    if _ASCEND_MAX_AUTOTUNE is None:
        from .ascend_autotune_runtime.autotuner import max_autotune as _fn
        _ASCEND_MAX_AUTOTUNE = _fn
    return _ASCEND_MAX_AUTOTUNE


# --- Proxies：模块 import 时直接替换，无需额外触发函数 ---
def _autotune_proxy(configs, key, **kwargs):
    if _USE_ASCEND:
        return _ascend_autotune_fn()(configs=configs, key=key, **kwargs)
    from triton.runtime.autotuner import autotune as _stock
    return _stock(configs=configs, key=key, **kwargs)


def _max_autotune_proxy(configs, key, kernel_type="mixcv", **kwargs):
    if _USE_ASCEND:
        return _ascend_max_autotune_fn()(
            configs=configs, key=key, kernel_type=kernel_type, **kwargs
        )
    return triton._default_max_autotune(configs=configs, key=key, **kwargs)


triton.autotune = _autotune_proxy
triton.max_autotune = _max_autotune_proxy


def hook_autotune_for_ascend():
    """Enable ascend mode（idempotent，仅 ascend backend 调用）。"""
    global _USE_ASCEND
    _USE_ASCEND = True


def unhook_autotune_for_ascend():
    """Restore stock-triton mode（测试/排障专用，非生产路径）。"""
    global _USE_ASCEND
    _USE_ASCEND = False
```

### 2.3 Proxy 安装时机

**必须** 在 `ascend_autotune_hooks.py` **首次 import** 时就把 `triton.autotune`
替换成 proxy。这样无论 `DICPDriver` 初始化在 `@triton.autotune()` 装饰之前还是之后，
proxy 都已经就位。

> 避免写成需要显式调用的 `_ensure_proxies_installed()` 函数——这会增加一个必须被
> 其他模块调用的时机约束，且更容易遗漏。

### 2.5 完整调用链

```
用户代码:
  @triton.autotune(configs=[], key=["M"])
  @triton.jit
  def kernel(...):
      ...

  kernel(...)

    |
    v
  triton.autotune (模块 import 时已被替换为 _autotune_proxy)
    |
    v
  _autotune_proxy(configs, key, **kwargs)  ──检查 _USE_ASCEND──┐
    |
    +-- False (默认 / unhook / 非 ascend backend):
    |   └─→ from triton.runtime.autotuner import autotune → 社区版 Autotuner
    |
    +-- True (hook 后 / ascend backend):
        └─→ ascend_autotune_runtime.autotuner.autotune() → AutoTilingTuner
                |
                v
          generate_key_and_configs()
                +-- _autoparse_axis_params()  → AST 解析
                +-- _gen_tile_configs()       → TileGenerator 生成候选
                |
                v
          _batch_bench()
                +-- _make_kernel_call(warmup=True) → ThreadPoolExecutor + AsyncCompileMode
                +-- do_bench_npu() / do_bench()     → NPU 基准测试
                |
                v
          cache[key] = best_config → fn.run()
```

### 2.6 为什么只在 backend=ascend 时替换？

```python
# backend/driver.py
elif backend == "ascend":
    ...
    from .ascend_autotune_hooks import hook_autotune_for_ascend
    hook_autotune_for_ascend()
```

- `_autotune_proxy` 在 `ascend_autotune_hooks` **首次 import 时就常驻**于 `triton` 模块
- proxy 内部检查 `_USE_ASCEND`：默认 `False`，所以**未 hook 前完全走 stock triton**
- `hook_autotune_for_ascend()` 仅在 DICPDriver `__init__` 的 ascend 分支被调用，立即切到 ascend 实现
- mlh / maca / nvidia 等 backend 不会触发 hook，`_USE_ASCEND` 永远为 `False`，行为与社区版一致
- `unhook_autotune_for_ascend()` 仅用于测试回滚，不是生产路径

## 3. 文件清单

### 3.1 新增模块：`backend/ascend_autotune_runtime/`

> 源自 triton-ascend/third_party/ascend/backend/runtime，import 路径调整为 DLCompiler (triton.backends.dicp_triton)。

| 文件 | 说明 | 调整 |
|------|------|------|
| `autoparser.py` (~1000 行) | AST 解析器：SplitAxesParser、TilingAxesParser、ReductionAxesParser、LowDimsAxesParser、PtrNumsParser | 无 |
| `tile_generator.py` (~530 行) | 基于 NPU 硬件约束生成 tiling 候选 | 无 |
| `utils.py` (~130 行) | 硬件常量 | 无 |
| `autotuner.py` (~1200 行) | AutoTilingTuner、max_autotune、BaseAutotuner | `do_bench_npu` → `triton.backends.dicp_triton.testing`；`is_compile_on_910_95` → `triton.backends.dicp_triton.utils` |
| `__init__.py` | 导出符号 | 新增 |

### 3.2 Hook 入口：`backend/ascend_autotune_hooks.py`

| 组件 | 说明 |
|------|------|
| `_USE_ASCEND` | 模块级 bool 标志，`True` 时 proxy 走 ascend 路径，`False` 时走 stock-triton |
| `_ascend_autotune_fn` / `_ascend_max_autotune_fn` | 懒加载辅助函数，首次 ascend 调用时才 import ascend 模块 |
| `_autotune_proxy` / `_max_autotune_proxy` | 常驻代理函数，直接检查 `_USE_ASCEND` 并分发。模块 import 时立即替换 `triton.autotune` / `triton.max_autotune` |
| `hook_autotune_for_ascend()` | 设置 `_USE_ASCEND = True`（幂等） |
| `unhook_autotune_for_ascend()` | 设置 `_USE_ASCEND = False`。**仅用于测试/排障**，不是生产路径 |

### 3.3 新增 NPU benchmark：`backend/testing.py`

> 源自 triton-ascend/third_party/ascend/backend/testing.py。
> 安装后映射到 `triton.backends.dicp_triton.testing`，供 `autotuner.py` import。

| 函数 | 说明 |
|------|------|
| `do_bench_npu(funcs, ...)` | NPU profiler benchmark：调用 `torch_npu.profiler.profile` 采集 `kernel_details.csv` |
| `_collect_prof_result(base_dir, funcs, ...)` | 从 profiler 结果目录解析 CSV，返回毫秒级耗时 |

| 调用点 | 说明 |
|--------|------|
| `autotuner._batch_bench()` | `_profile_bench` 分支，对比 config 时调用 |
| `autotuner._profile()` | `auto_profile_dir` 有值时调用，保存详细 profiling 结果 |

### 3.4 其他修改

| 文件 | 调整 |
|------|------|
| `backend/driver.py` (+2 行) | ascend 分支调用 `hook_autotune_for_ascend()` |
| `patch/triton/python_triton___init___py.patch` | `triton/__init__.py` 补充 `max_autotune` fallback + `_default_max_autotune` |

## 4. AutoTilingTuner 数据流（Ascend 策略）

```
装饰阶段:
  @triton.autotune(configs=[], key={"x":"M","y":"N"}, hints={...})
  → _autotune_proxy() → _ascend_autotune_fn()
      → ascend_autotune_runtime.autotuner.autotune()
          → 返回 AutoTilingTuner 实例（保存 hints/keys）

运行时 (首次调用，key miss):
  AutoTilingTuner.run()
  → generate_key_and_configs()
     → _autoparse_axis_params()
        → AST 解析 (SplitAxesParser/TilingAxesParser/...)
           - pid * BLOCK_M → split axis "x"
           - tl.arange(0, BLOCK_K) → tiling axis "k"
        → _gen_tile_configs()
           → TileGenerator(kernel_meta)
              → descend_split_tiling()
                 - UB/RF 大小过滤、core 数约束
                 - 生成合法 tiling config 列表
     → configs = gen_configs + user_configs
  → _batch_bench()
     +-- _make_kernel_call(warmup=True) → ThreadPoolExecutor + AsyncCompileMode (并行编译)
     +-- do_bench_npu() / do_bench() (NPU 基准测试)
  → cache[key] = best_config

运行时 (cache hit):
  → 直接取 best_config，跳过 AST 和 benchmark
  → fn.run(*args, **best_config.kwargs, **kwargs)
```

## 5. max_autotune 数据流（Ascend 策略）

```
@triton.max_autotune(configs=[Config(BLOCK=64)], key=["M"],
                     kernel_type="mixcv", enable_hivm_auto_cv_balance=[True, False])
  → _max_autotune_proxy() → _ascend_max_autotune_fn()
      → get_max_configs()
          [Config(BLOCK=64)] × {enable_hivm_auto_cv_balance: [True, False]}
          = 2 个扩展后的 Config
      → autotune(configs=expanded, key=key, ...) → AutoTilingTuner
          → 运行阶段同上

Config.kwargs (含编译选项)
  → JITFunction.run() → binder → options_dict
  → DICPBackend.parse_options() → NPUOptions
  → metadata = {**options.__dict__}
  → linalg_to_bin_*() → bishengir-compile flags
```

## 6. 与 triton-ascend 行为对照

| 方面 | triton-ascend | DLCompiler 本方案 |
|------|---------------|-------------------|
| `configs=[]` 自动 tiling | AutoTilingTuner + TileGenerator | 同（直接复用源码） |
| `configs=[...]` 基准测试 | `_batch_bench` + `_make_kernel_call` | 同 |
| 并行编译 | ThreadPoolExecutor + AsyncCompileMode | 同 |
| NPU benchmark | `do_bench_npu` | 同（从 dicp_triton.testing 导入） |
| compile-option 搜索 | `get_max_configs` + `BaseAutotuner` | 同 |
| triton 入口替换 | `triton.autotune = autotune` (ascend 硬编码) | `triton.autotune = _autotune_proxy`，proxy 查看 `_USE_ASCEND` 标志 |
| 其他 backend 影响 | ascend 版替换全局 | Hook 只在 ascend 分支触发，其他 backend 天然不变 |

## 7. 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TRITON_AUTOTUNE_PARALLEL_COMPILE` | `"1"` | 是否启用并行编译（仅对 JITFunction） |
| `TRITON_PRINT_AUTOTUNING` | `None` | 打印 autotune 结果 |
| `TRITON_BENCH_METHOD` | `"default"` | `"npu"` 时使用 NPU profiler |

## 8. 测试要点

1. **backend=ascend**：`@triton.autotune(configs=[])` 触发 AST 自动 tiling；`@triton.max_autotune` 展开编译选项
2. **backend=mlu/maca/nvidia**：autotune 行为与社区版一致
3. **幂等性**：`hook_autotune_for_ascend()` 调用多次安全
4. **可逆性**：`unhook_autotune_for_ascend()` 恢复原始行为
5. **编译失败容错**：单个 config 编译失败不影响其他 config

## 9. Patch 文件

| Patch 文件 | 目标 | 说明 |
|-----------|------|------|
| `patch/triton/python_triton___init___py.patch` | `python/triton/__init__.py` | 补充 `max_autotune` fallback + `_default_max_autotune` |

命名规范：`python/triton/<path>` → `python_triton_<path_分隔符为_>.patch`
