// ============================================================================
// DSL Custom Op — add (multi-dtype: int32 / fp32 / fp16)
//
// 作用: 在一个 bitcode 中提供三种 dtype 的 vadd C 接口，供 MLIR 调用。
// 编译: bash compile.sh → add.aiv.bc
// 符号:
//   _mlir_ciface_custom_add_int32  (对应 dl custom_add_int32)
//   _mlir_ciface_custom_add_fp32   (对应 dl custom_add_fp32)
//   _mlir_ciface_custom_add_fp16   (对应 dl custom_add_fp16)
//
// 共享位码库: ops/skills/triton/triton-dsl-custom-op/bitcode_lib/
// ============================================================================

#define __aiv__ [aicore]
#define INTRINSIC_NO_ARGS(NAME) NAME()
#define INTRINSIC(NAME, ...) NAME(__VA_ARGS__)

// MLIR memref 结构：表示一块连续内存（指针 + offset + shape + strides）
template <typename T, size_t Dim>
struct memref_t {
  T *allocated;              // 分配基址
  T *aligned;                // 对齐后的有效起始地址
  int64_t offset;            // 元素偏移量
  int64_t sizes[Dim];        // 各维度长度
  int64_t strides[Dim];      // 各维度步长
};

// vadd 指令参数结构
template <size_t OPERANUM, typename SRC_T, typename DST_T = SRC_T>
struct intrin_args {
  __ubuf__ DST_T *dst;                     // 输出指针
  __ubuf__ SRC_T *src[OPERANUM];           // 输入指针数组
  SRC_T scalar;                            // 标量值（未使用）
  uint64_t repeat;                         // 重复次数
  uint16_t dst_block_stride;               // block 内步长
  uint16_t src_block_stride[OPERANUM];     // 输入 block 内步长
  uint16_t dst_repeat_stride;              // repeat 间步长
  uint16_t src_repeat_stride[OPERANUM];    // 输入 repeat 间步长
};

// vadd 模板函数：逐元素向量加法
template <typename SRC_TYPE, typename DST_TYPE = SRC_TYPE>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vadd_intrin(intrin_args<2, SRC_TYPE, DST_TYPE> args) {
#define ELTWISE_VV_ARGS                                                        \
  args.dst, args.src[0], args.src[1], args.repeat, args.dst_block_stride,      \
      args.src_block_stride[0], args.src_block_stride[1],                      \
      args.dst_repeat_stride, args.src_repeat_stride[0],                       \
      args.src_repeat_stride[1]

  // vadd(dst, src0, src1, repeat, dst_bs, src0_bs, src1_bs, dst_rs, src0_rs, src1_rs)
  INTRINSIC(vadd, ELTWISE_VV_ARGS);
}

// vadd 调用包装：处理连续访问的公共逻辑
template <typename T>
__aiv__ __attribute__((always_inline)) void
vadd_impl(memref_t<__ubuf__ T, 1> *src0,
          memref_t<__ubuf__ T, 1> *src1,
          memref_t<__ubuf__ T, 1> *dst) {

  uint16_t block_stride = 1;
  uint16_t repeat_stride = 8;

  auto new_src0_ptr = src0->aligned + src0->offset;
  auto new_src1_ptr = src1->aligned + src1->offset;
  auto dst_ptr = dst->aligned + dst->offset;

  // 设置向量掩码（处理边界）
  INTRINSIC_NO_ARGS(set_mask_count);
  const int64_t n = dst->sizes[0];
  INTRINSIC(set_vector_mask, 0, n);

  // 调用 vadd 指令
  vector_eltwise_vadd_intrin<T>(
      intrin_args<2, T>{dst_ptr,
                        {new_src0_ptr, new_src1_ptr},
                        0,       // scalar (unused)
                        1,       // repeat = 1（单次执行，全部元素由 mask 覆盖）
                        block_stride,
                        {block_stride, block_stride},
                        repeat_stride,
                        {repeat_stride, repeat_stride}});

  // 恢复掩码
  INTRINSIC_NO_ARGS(set_mask_norm);
}

// ============================================================================
// MLIR 可调用的 C 接口（三种 dtype）
//
// Python 侧 symbol 命名规则:
//   str(tl.int32)    → "int32"  → symbol = "custom_add_int32"
//   str(tl.float32)  → "fp32"   → symbol = "custom_add_fp32"
//   str(tl.float16)  → "fp16"   → symbol = "custom_add_fp16"
//
// MLIR 自动添加 _mlir_ciface_ 前缀:
//   Python "custom_add_int32" → C _mlir_ciface_custom_add_int32
//   Python "custom_add_fp32"  → C _mlir_ciface_custom_add_fp32
//   Python "custom_add_fp16"  → C _mlir_ciface_custom_add_fp16
// ============================================================================

extern "C" {

__aiv__ __attribute__((always_inline)) void _mlir_ciface_custom_add_int32(
    memref_t<__ubuf__ int32_t, 1> *src0,
    memref_t<__ubuf__ int32_t, 1> *src1,
    memref_t<__ubuf__ int32_t, 1> *dst) {
  vadd_impl(src0, src1, dst);
}

__aiv__ __attribute__((always_inline)) void _mlir_ciface_custom_add_fp32(
    memref_t<__ubuf__ float, 1> *src0,
    memref_t<__ubuf__ float, 1> *src1,
    memref_t<__ubuf__ float, 1> *dst) {
  vadd_impl(src0, src1, dst);
}

__aiv__ __attribute__((always_inline)) void _mlir_ciface_custom_add_fp16(
    memref_t<__ubuf__ __fp16, 1> *src0,
    memref_t<__ubuf__ __fp16, 1> *src1,
    memref_t<__ubuf__ __fp16, 1> *dst) {
  vadd_impl(src0, src1, dst);
}

}  // extern "C"
