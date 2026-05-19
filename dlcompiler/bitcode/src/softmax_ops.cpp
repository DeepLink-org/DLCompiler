// ============================================================================
// DSL Custom Op — softmax 相关运算 (fp32)
//
// 包含用于 FlashAttention 的 vector 操作:
//   vexp:  y = exp(x)              单目
//   vdiv:  z = x / y               双目
//   vsub:  z = x - y               双目
//   vmul:  z = x * y               双目
//
// 编译: bash compile.sh → softmax_ops.aiv.bc
// 符号:
//   _mlir_ciface_custom_vexp_fp32
//   _mlir_ciface_custom_vdiv_fp32
//   _mlir_ciface_custom_vsub_fp32
//   _mlir_ciface_custom_vmul_fp32
//
// 共享位码库: ops/skills/triton/triton-dsl-custom-op/bitcode_lib/
// ============================================================================

#define __aiv__ [aicore]
#define INTRINSIC_NO_ARGS(NAME) NAME()
#define INTRINSIC(NAME, ...) NAME(__VA_ARGS__)

template <typename T, size_t Dim>
struct memref_t {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[Dim];
  int64_t strides[Dim];
};

// ============================================================================
// 公共包装: 设置 mask → 调用 intrinsic → 恢复 mask
// 所有操作都是连续访问: block_stride=1, repeat_stride=8
// ============================================================================

// 单目操作包装 (vexp)
template <typename T>
__aiv__ __attribute__((always_inline)) void
unary_op_impl(void (*intrin)(__ubuf__ T *, __ubuf__ T *, uint64_t,
                              uint16_t, uint16_t, uint16_t, uint16_t),
              memref_t<__ubuf__ T, 1> *src,
              memref_t<__ubuf__ T, 1> *dst) {
  auto s = src->aligned + src->offset;
  auto d = dst->aligned + dst->offset;
  int64_t n = dst->sizes[0];

  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0, n);

  // 单目: (dst, src, repeat, dst_bs, src_bs, dst_rs, src_rs)
  intrin(d, s, 1, 1, 1, 8, 8);

  INTRINSIC_NO_ARGS(set_mask_norm);
}

// 双目操作包装 (vdiv, vsub, vmul)
template <typename T>
__aiv__ __attribute__((always_inline)) void
binary_op_impl(void (*intrin)(__ubuf__ T *, __ubuf__ T *, __ubuf__ T *,
                               uint64_t, uint16_t, uint16_t, uint16_t,
                               uint16_t, uint16_t, uint16_t),
               memref_t<__ubuf__ T, 1> *src0,
               memref_t<__ubuf__ T, 1> *src1,
               memref_t<__ubuf__ T, 1> *dst) {
  auto s0 = src0->aligned + src0->offset;
  auto s1 = src1->aligned + src1->offset;
  auto d  = dst->aligned + dst->offset;
  int64_t n = dst->sizes[0];

  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0, n);

  // 双目: (dst, src0, src1, repeat, dst_bs, src0_bs, src1_bs, dst_rs, src0_rs, src1_rs)
  intrin(d, s0, s1, 1, 1, 1, 1, 8, 8, 8);

  INTRINSIC_NO_ARGS(set_mask_norm);
}

// ============================================================================
// MLIR 可调用的 C 接口
//
// Python 侧 symbol 名:
//   "custom_vexp_fp32"  → C _mlir_ciface_custom_vexp_fp32
//   "custom_vdiv_fp32"  → C _mlir_ciface_custom_vdiv_fp32
//   "custom_vsub_fp32"  → C _mlir_ciface_custom_vsub_fp32
//   "custom_vmul_fp32"  → C _mlir_ciface_custom_vmul_fp32
// ============================================================================

extern "C" {

// --- vexp: y = exp(x) ---
__aiv__ __attribute__((always_inline)) void _mlir_ciface_custom_vexp_fp32(
    memref_t<__ubuf__ float, 1> *src,
    memref_t<__ubuf__ float, 1> *dst) {
  // vexp(dst, src, repeat, dst_bs, src_bs, dst_rs, src_rs)
  auto s = src->aligned + src->offset;
  auto d = dst->aligned + dst->offset;
  int64_t n = dst->sizes[0];
  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0, n);
  INTRINSIC(vexp, d, s, 1, 1, 1, 8, 8);
  INTRINSIC_NO_ARGS(set_mask_norm);
}

// --- vdiv: z = x / y ---
__aiv__ __attribute__((always_inline)) void _mlir_ciface_custom_vdiv_fp32(
    memref_t<__ubuf__ float, 1> *src0,
    memref_t<__ubuf__ float, 1> *src1,
    memref_t<__ubuf__ float, 1> *dst) {
  auto s0 = src0->aligned + src0->offset;
  auto s1 = src1->aligned + src1->offset;
  auto d  = dst->aligned + dst->offset;
  int64_t n = dst->sizes[0];
  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0, n);
  INTRINSIC(vdiv, d, s0, s1, 1, 1, 1, 1, 8, 8, 8);
  INTRINSIC_NO_ARGS(set_mask_norm);
}

// --- vsub: z = x - y ---
__aiv__ __attribute__((always_inline)) void _mlir_ciface_custom_vsub_fp32(
    memref_t<__ubuf__ float, 1> *src0,
    memref_t<__ubuf__ float, 1> *src1,
    memref_t<__ubuf__ float, 1> *dst) {
  auto s0 = src0->aligned + src0->offset;
  auto s1 = src1->aligned + src1->offset;
  auto d  = dst->aligned + dst->offset;
  int64_t n = dst->sizes[0];
  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0, n);
  INTRINSIC(vsub, d, s0, s1, 1, 1, 1, 1, 8, 8, 8);
  INTRINSIC_NO_ARGS(set_mask_norm);
}

// --- vmul: z = x * y ---
__aiv__ __attribute__((always_inline)) void _mlir_ciface_custom_vmul_fp32(
    memref_t<__ubuf__ float, 1> *src0,
    memref_t<__ubuf__ float, 1> *src1,
    memref_t<__ubuf__ float, 1> *dst) {
  auto s0 = src0->aligned + src0->offset;
  auto s1 = src1->aligned + src1->offset;
  auto d  = dst->aligned + dst->offset;
  int64_t n = dst->sizes[0];
  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0, n);
  INTRINSIC(vmul, d, s0, s1, 1, 1, 1, 1, 8, 8, 8);
  INTRINSIC_NO_ARGS(set_mask_norm);
}

}  // extern "C"
