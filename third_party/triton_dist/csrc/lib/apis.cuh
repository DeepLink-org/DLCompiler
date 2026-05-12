/*
 * Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include <driver_types.h>
#include <torch/all.h>

namespace distributed {

namespace ops {

void moe_ag_scatter_align_block_size_parallel_op(
    torch::Tensor topk_ids, int32_t topk, int32_t num_experts,
    int32_t num_ranks, int32_t num_tokens_per_rank, int32_t block_size,
    torch::Tensor sorted_token_ids, torch::Tensor experts_ids,
    torch::Tensor block_barrier_ids, torch::Tensor rank_block_num,
    torch::Tensor num_tokens_post_pad, intptr_t moe_stream);

void moe_ag_scatter_align_block_size_op(
    torch::Tensor topk_ids, int32_t num_experts, int32_t num_iterations,
    int32_t num_tokens_per_iteration, int32_t block_size,
    torch::Tensor sorted_token_ids, torch::Tensor experts_ids,
    torch::Tensor block_barrier_ids, torch::Tensor rank_block_num,
    torch::Tensor num_tokens_post_pad, intptr_t moe_stream);
} // namespace ops
} // namespace distributed
