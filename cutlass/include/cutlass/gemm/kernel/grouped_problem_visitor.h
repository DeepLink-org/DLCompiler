/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Base scheduler for grouped problems
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"

#include "cutlass/fast_math.h"
#include "cutlass/barrier.h"

#define cudaAssert(condition) \
  if (!(condition)){ printf("Assertion %s failed!\n", #condition); asm("trap;"); }


#define ASSERT(condition) do { \
    if (!(condition)) { \
        std::cerr << "Assertion failed: " #condition " at " \
                  << __FILE__ << ":" << __LINE__ << std::endl; \
        abort(); \
    } \
} while(0)

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Enumerated type describing the type of scheduling to perform for the ProblemVisitor
enum class GroupScheduleMode {
  // Perform all scheduling on device
  kDeviceOnly,
  // Precompute on the host the full sequence of problems to access
  kHostPrecompute,

  mixedStreamK,
};

/// Visitor class to abstract away the algorithm for iterating over tiles
template <typename ProblemSizeHelper,
          typename ThreadblockShape_>
struct BaseGroupedProblemVisitor {
  using ThreadblockShape = ThreadblockShape_;

  struct ProblemInfo {
    static int32_t const kNoPrefetchEntry = -1;
    int32_t problem_idx;
    int32_t problem_start;

    CUTLASS_DEVICE
    ProblemInfo() : problem_idx(kNoPrefetchEntry), problem_start(kNoPrefetchEntry) {}

    CUTLASS_DEVICE
    ProblemInfo(int32_t problem_idx_, int32_t problem_start_) :
      problem_idx(problem_idx_), problem_start(problem_start_) {}
  };

  struct Params {
    cutlass::gemm::GemmCoord const *problem_sizes;
    int32_t                         problem_count;
    void                      *workspace;
    int32_t                         tile_count;

    //
    // Methods
    //

    /// Ctor
    CUTLASS_HOST_DEVICE
    Params(): problem_sizes(nullptr), problem_count(0), workspace(nullptr), tile_count(0) { }

    /// Ctor
    CUTLASS_HOST_DEVICE
    Params(
      cutlass::gemm::GemmCoord const *problem_sizes,
      int32_t                         problem_count,
      void                      *workspace = nullptr,
      int32_t                         tile_count = 0
    ):
      problem_sizes(problem_sizes),
      problem_count(problem_count),
      workspace(workspace),
      tile_count(tile_count)
    {}

  };

  Params params;
  int32_t tile_idx;
  int32_t problem_tile_start;
  int32_t problem_idx;

  //
  // Methods
  //
  CUTLASS_DEVICE
  BaseGroupedProblemVisitor(
    Params &params_,
    int32_t block_idx
  ):
  params(params_),
  tile_idx(block_idx),
  problem_tile_start(0),
  problem_idx(0)
  {}

  /// Get the grid shape
  CUTLASS_HOST_DEVICE
  static cutlass::gemm::GemmCoord grid_shape(const cutlass::gemm::GemmCoord& problem) {
    return ProblemSizeHelper::grid_shape(problem);
  }

  /// Gets the global tile index
  CUTLASS_HOST_DEVICE
  int32_t tile_index() const {
    return tile_idx;
  }

  /// Gets the index of the problem
  CUTLASS_HOST_DEVICE
  int32_t problem_index() const {
    return problem_idx;
  }

  CUTLASS_HOST_DEVICE
  int32_t threadblock_idx() const {
    return tile_idx - problem_tile_start;
  }

  CUTLASS_DEVICE
  void advance(int32_t grid_size) {
    tile_idx += grid_size;
  }

  CUTLASS_HOST_DEVICE
  static void possibly_transpose_problem(cutlass::gemm::GemmCoord& problem) {
    ProblemSizeHelper::possibly_transpose_problem(problem);
  }

  /// Returns the problem size for the current problem
  CUTLASS_HOST_DEVICE
  cutlass::gemm::GemmCoord problem_size() const {
    GemmCoord problem = params.problem_sizes[problem_idx];
    ProblemSizeHelper::possibly_transpose_problem(problem);
    return problem;
  }

  CUTLASS_HOST_DEVICE
  static int32_t tile_count(const cutlass::gemm::GemmCoord& grid) {
    return ProblemSizeHelper::tile_count(grid);
  }

  static int32_t group_tile_count(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr, int32_t problem_count) {
    int32_t total_tiles = 0;
    for (int32_t i = 0; i < problem_count; ++i) {
      auto problem = host_problem_sizes_ptr[i];
      possibly_transpose_problem(problem);
      auto grid = grid_shape(problem);
      total_tiles += tile_count(grid);
    }

    return total_tiles;
  }

  static size_t get_workspace_size_streamk(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                                   int32_t problem_count,
                                   int32_t block_count,
                                   size_t kWorkspaceBytesPerBlock,
                                   //
                                  int num_sms,
                                  int sm_occupancy,
                                  int mma_shape_k,
                                  GemmCoord thread_block_shape
                                  )
                                  {return 0;}
  static void host_precompute_streamk(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                              int32_t problem_count,
                              int32_t block_count,
                                   size_t kWorkspaceBytesPerBlock,
                              //
                              int num_sms,
                              int sm_occupancy,
                                  int mma_shape_k,
                              GemmCoord thread_block_shape,
                              //
                              void* host_workspace_ptr) {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ProblemSizeHelper,
  typename ThreadblockShape,
  GroupScheduleMode GroupScheduleMode_,
  int PrefetchTileCount,
  int ThreadCount
>
struct GroupedProblemVisitor;

/////////////////////////////////////////////////////////////////////////////////////////////////
// ProblemVisitor that performs all scheduling on device
//
template <typename ProblemSizeHelper,
          typename ThreadblockShape,
          int PrefetchTileCount,
          int ThreadCount>
struct GroupedProblemVisitor<ProblemSizeHelper,
                             ThreadblockShape,
                             GroupScheduleMode::kDeviceOnly,
                             PrefetchTileCount,
                             ThreadCount>: public BaseGroupedProblemVisitor<ProblemSizeHelper, ThreadblockShape> {
  using Base = BaseGroupedProblemVisitor<ProblemSizeHelper, ThreadblockShape>;
  using Params = typename Base::Params;
  static int const kThreadCount = ThreadCount;
  static int const kRequiresPrecomputation = 0;
  static int const kThreadsPerWarp = 32;

  struct SharedStorage {};

  // Final tile of the problem loaded by this thread. Each thread will hold
  // a separate value.
  int32_t problem_ending_tile;

  SharedStorage &shared_storage;

  //
  // Methods
  //
  CUTLASS_DEVICE
  GroupedProblemVisitor(
    Params &params_,
    SharedStorage &shared_storage_,
    int32_t block_idx
  ): Base(params_, block_idx),
  problem_ending_tile(0),
  shared_storage(shared_storage_)
  {
    this->problem_idx = -1 * kThreadsPerWarp;
    this->problem_tile_start = 0;

    // if ((block_idx == 0 || block_idx == 1) && threadIdx.x == 0)
    //   printf("tile_count: %d, tile_idx: %d, problem_tile_start: %d, problem_idx: %d, problem_ending_tile: %d\n", this->params.tile_count, this->tile_idx, this->problem_tile_start, this->problem_idx, this->problem_ending_tile);

  }

  CUTLASS_DEVICE
  bool next_tile() {
    // Check whether the tile to compute is within the range of the current problem.
    int32_t problem_tile_end = __shfl_sync(0xffffffff, problem_ending_tile, this->problem_idx % kThreadsPerWarp);
    if (this->tile_idx < problem_tile_end) {
      return true;
    }

    // Check whether the tile to compute is within the current group of problems fetched by the warp.
    // The last tile for this group is the final tile of the problem held by the final thread in the warp.
    int32_t group_tile_end = __shfl_sync(0xffffffff, problem_ending_tile, kThreadsPerWarp-1);

    // Keep the starting problem for this group in `problem_idx`. This is done to reduce
    // register pressure. The starting problem for this group is simply the first problem
    // in the group most recently fetched by the warp.
    int32_t &group_problem_start = this->problem_idx;
    group_problem_start = (this->problem_idx / kThreadsPerWarp) * kThreadsPerWarp;

    // Keep the starting tile for this group in `problem_tile_start`. This is done to reduce
    // register pressure.
    int32_t &group_tile_start = this->problem_tile_start;

    // Each thread in the warp processes a separate problem to advance until
    // reaching a problem whose starting tile is less less than tile_idx.
    while (group_tile_end <= this->tile_idx) {
      group_problem_start += kThreadsPerWarp;
      if (group_problem_start > this->params.problem_count) {
        return false;
      }

      // Since `group_tile_start` is a reference to `this->problem_tile_start`, this
      // also sets `this->problem_tile_start`. The fact that `this->problem_tile_start`
      // is also set here is used later in `next_tile`.
      group_tile_start = group_tile_end;

      int lane_idx = threadIdx.x % kThreadsPerWarp;
      int32_t lane_problem = group_problem_start + lane_idx;

      // Compute the number of tiles in the problem assigned to each thread.
      problem_ending_tile = 0;
      if (lane_problem < this->params.problem_count) {
        cutlass::gemm::GemmCoord problem = this->params.problem_sizes[lane_problem];
        this->possibly_transpose_problem(problem);
        cutlass::gemm::GemmCoord grid = this->grid_shape(problem);
        problem_ending_tile = this->tile_count(grid);
      }

      // Compute a warp-wide inclusive prefix sum to compute the ending tile index of
      // each thread's problem.
      CUTLASS_PRAGMA_UNROLL
      for (int i = 1; i < kThreadsPerWarp; i <<= 1) {
        int32_t val = __shfl_up_sync(0xffffffff, problem_ending_tile, i);
        if (lane_idx >= i) {
          problem_ending_tile += val;
        }
      }

      // The total tile count for this group is now in the final position of the prefix sum
      int32_t tiles_in_group = __shfl_sync(0xffffffff, problem_ending_tile, kThreadsPerWarp-1);

      problem_ending_tile += group_tile_start;
      group_tile_end += tiles_in_group;
    }

    // The next problem to process is the first one that does not have ending tile position
    // that is greater than or equal to tile index.
    int32_t problem_idx_in_group =
        __popc(__ballot_sync(0xffffffff, problem_ending_tile <= this->tile_idx));

    this->problem_idx = group_problem_start + problem_idx_in_group;

    // The starting tile for this problem is the ending tile of the previous problem. In cases
    // where `problem_idx_in_group` is the first problem in the group, we do not need to reset
    // `problem_tile_start`, because it is set to the previous group's ending tile in the while
    // loop above.
    if (problem_idx_in_group > 0) {
      this->problem_tile_start = __shfl_sync(0xffffffff, problem_ending_tile, problem_idx_in_group - 1);
    }

    // if ((blockIdx.x == 0) && threadIdx.x == 0)
    //   printf("[Next] tiled_idx: %d, problem_tile_start: %d, problem_tile_end: %d, group_tile_start: %d, group_tile_end: %d, problem_idx: %d, problem_index_in_group: %d\n", this->tile_idx, this->problem_tile_start, problem_tile_end, group_tile_start, group_tile_end, this->problem_idx, problem_idx_in_group);

    return true;
  }

  static size_t get_workspace_size(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                                   int32_t problem_count,
                                   int32_t block_count) {
    return 0;
  }

  static void host_precompute(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                              int32_t problem_count,
                              int32_t block_count,
                              void* host_workspace_ptr) {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// Precomputes schedule on host and prefetches into shared memory
//
template <typename ProblemSizeHelper,
          typename ThreadblockShape,
          int PrefetchTileCount,
          int ThreadCount>
struct GroupedProblemVisitor<ProblemSizeHelper,
                             ThreadblockShape,
                             GroupScheduleMode::kHostPrecompute,
                             PrefetchTileCount,
                             ThreadCount> : public BaseGroupedProblemVisitor<ProblemSizeHelper, ThreadblockShape> {
  static_assert(PrefetchTileCount > 0,
                "GroupedProblemVisitor with GroupScheduleMode `kHostPrecompute` currently requires prefetching to shared memory");

  using Base = BaseGroupedProblemVisitor<ProblemSizeHelper, ThreadblockShape>;
  using Params = typename Base::Params;
  using ProblemInfo = typename Base::ProblemInfo;
  static int const kRequiresPrecomputation = 1;

  static int const kPrefetchTileCount = PrefetchTileCount;
  static int const kThreadCount = ThreadCount;

  struct SharedStorage {
    // Sequence of problem IDs and starting tiles to compute
    cutlass::Array<ProblemInfo, kPrefetchTileCount> prefetched_problems;
  };

  int32_t tiles_computed;
  int32_t iterations_per_block;
  int32_t block_load_start;
  SharedStorage &shared_storage;
  ProblemInfo const *problem_info_ptr;

  //
  // Methods
  //
  CUTLASS_DEVICE
  GroupedProblemVisitor(
    Params &params_,
    SharedStorage &shared_storage_,
    int32_t block_idx
  ): Base(params_, block_idx),
  tiles_computed(0),
  shared_storage(shared_storage_),
  problem_info_ptr(reinterpret_cast<ProblemInfo const*>(params_.workspace))
  {
    iterations_per_block = (params_.tile_count - 1 + gridDim.x) / gridDim.x;
    block_load_start = iterations_per_block * block_idx;

    // if ((block_idx == 0 || block_idx == 1 || block_idx==127) && threadIdx.x == 0)
    //   printf("tile_count: %d, tiles_computed: %d, tile_idx: %d, problem_tile_start: %d, problem_idx: %d, iterations_per_block: %d, block_load_start: %d, kPrefetchTileCount: %d, kThreadCount: %d\n", params_.tile_count, tiles_computed, this->tile_idx, this->problem_tile_start, this->problem_idx, iterations_per_block, block_load_start, kPrefetchTileCount, kThreadCount);

    // Start prefetching the first set of tiles to compute
    prefetch_tiles();
  }

  CUTLASS_DEVICE
  bool next_tile() {
    if (this->tile_idx >= this->params.tile_count) {
      return false;
    }

    int32_t prefetch_idx = (tiles_computed % kPrefetchTileCount);
    if (prefetch_idx == 0) {
      // Ensure all previous stores to shared memory have been completed
      __syncthreads();
    }

    auto problem_info = shared_storage.prefetched_problems[prefetch_idx];
    ++tiles_computed;

    if ((tiles_computed % kPrefetchTileCount) == 0) {
      // Begin prefetching next set of tiles. Synchronize first to ensure that
      // we don't overwrite the current buffer while someone else is using it.
      __syncthreads();
      prefetch_tiles();
    }

    this->problem_idx = problem_info.problem_idx;
    this->problem_tile_start = problem_info.problem_start;

    //if ((blockIdx.x == 0) && threadIdx.x == 0)
    //  printf("[Next] tiled_idx: %d, prefetch_idx: %d, tiles_computed: %d, problem_idx: %d, problem_tile_start: %d\n", this->tile_idx, prefetch_idx, tiles_computed, this->problem_idx, this->problem_tile_start);

    return true;
  }

  static size_t get_workspace_size(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                                   int32_t problem_count,
                                   int32_t block_count) {
    int32_t total_tiles = Base::group_tile_count(host_problem_sizes_ptr, problem_count);
    int32_t entries_per_block = ((total_tiles - 1 + block_count) / block_count);
    return sizeof(ProblemInfo) * entries_per_block * block_count;
  }
#if !defined(__CUDACC_RTC__)
  static void host_precompute(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                              int32_t problem_count,
                              int32_t block_count,
                              void* host_workspace_ptr) {
    ProblemInfo* host_problem_info_ptr = reinterpret_cast<ProblemInfo*>(host_workspace_ptr);
    int32_t total_tiles = Base::group_tile_count(host_problem_sizes_ptr, problem_count);
    int32_t entries_per_block = (total_tiles - 1 + block_count) / block_count;

    int tile = 0;
    int start_tile = 0;
    for (int p_idx = 0; p_idx < problem_count; ++p_idx) {
      auto problem = host_problem_sizes_ptr[p_idx];
      Base::possibly_transpose_problem(problem);
      auto grid = Base::grid_shape(problem);
      int tiles = Base::tile_count(grid);
      ProblemInfo problem_info(p_idx, start_tile);
      for (int i = 0; i < tiles; ++i, ++tile) {
        host_problem_info_ptr[(entries_per_block * (tile % block_count)) + (tile / block_count)] = problem_info;
      }
      start_tile += tiles;
    }
  }
#endif
private:
  CUTLASS_DEVICE
  void prefetch_tiles() {
    CUTLASS_PRAGMA_UNROLL
    for (int32_t i = 0; i < kPrefetchTileCount; i += kThreadCount) {
      int32_t offset = threadIdx.x + i;
      if (offset < kPrefetchTileCount && (tiles_computed + offset < iterations_per_block)) {
        shared_storage.prefetched_problems[offset] = problem_info_ptr[block_load_start + tiles_computed + offset];
      }
    }
  }
};



/////////////////////////////////////////////////////////////////////////////////////////////////
// mixed streamK
//
template <typename ProblemSizeHelper,
          typename ThreadblockShape,
          int PrefetchTileCount,
          int ThreadCount>
struct GroupedProblemVisitor<ProblemSizeHelper,
                             ThreadblockShape,
                             GroupScheduleMode::mixedStreamK,
                             PrefetchTileCount,
                             ThreadCount> : public BaseGroupedProblemVisitor<ProblemSizeHelper, ThreadblockShape> {
  static_assert(PrefetchTileCount > 0,
                "mixedStreamK currently requires prefetching to shared memory");

  using Base = BaseGroupedProblemVisitor<ProblemSizeHelper, ThreadblockShape>;
  using Params = typename Base::Params;
  using ProblemInfo = typename Base::ProblemInfo;

  static int const kRequiresPrecomputation = 2;

  static int const kPrefetchTileCount = PrefetchTileCount;
  static int const kThreadCount = ThreadCount;

  // customized with sk info
  struct skInfo {
    int sk_regions;
    int dp_blocks;
    int dp_tiles;
    int sk_blocks;
    //int sk_blocks_per_region;
    int sk_tiles;
    int sk_waves;
    int sk_iters_per_normal_block;
    int sk_big_blocks_per_region;
    int iters_per_tile;
    int problem_size_k;
    int mma_shape_k;
    GemmCoord thread_block_shape;

    //
    FastDivmod div_mod_sk_iters_per_normal_block;
    FastDivmod div_mod_sk_iters_per_big_block;
    FastDivmod div_mod_sk_iters_per_region;
    FastDivmod div_mod_iters_per_tile;
    //
      size_t partials_workspace_bytes;
      size_t barrier_workspace_bytes;
    size_t sk_workspace_size ;
    size_t tile_workspace_size ;

    // std::unordered_map<int, std::tuple<int, int, int>> tile_idx_to_offset{};
    int entries_per_block;  // dp entries per block

    CUTLASS_DEVICE
    skInfo(){}

    CUTLASS_DEVICE
    skInfo(
    int sk_regions,
    int dp_blocks,
    int dp_tiles,
    int sk_blocks,
    //int sk_blocks_per_region,
    int sk_tiles,
    int sk_waves,
    int sk_iters_per_normal_block,
    int sk_big_blocks_per_region,
    int iters_per_tile,
    int problem_size_k,
    int mma_shape_k,
    GemmCoord thread_block_shape,

    //
    FastDivmod div_mod_sk_iters_per_normal_block,
    FastDivmod div_mod_sk_iters_per_big_block,
    FastDivmod div_mod_sk_iters_per_region,
    FastDivmod div_mod_iters_per_tile,
    //
      size_t partials_workspace_bytes,
      size_t barrier_workspace_bytes,
    size_t sk_workspace_size ,
    size_t tile_workspace_size ,

    int entries_per_block
    ) : sk_regions(sk_regions)
        ,dp_blocks(dp_blocks)
        ,dp_tiles(dp_tiles)
        ,sk_blocks(sk_blocks)
        //,sk_blocks_per_region(sk_blocks_per_region)
        ,sk_tiles(sk_tiles)
        ,sk_waves(sk_waves)
        ,sk_iters_per_normal_block(sk_iters_per_normal_block)
        ,sk_big_blocks_per_region(sk_big_blocks_per_region)
        ,iters_per_tile(iters_per_tile)
        ,problem_size_k(problem_size_k)
        ,mma_shape_k(mma_shape_k)
        ,thread_block_shape(thread_block_shape)
        ,div_mod_sk_iters_per_normal_block(div_mod_sk_iters_per_normal_block)
        ,div_mod_sk_iters_per_big_block(div_mod_sk_iters_per_big_block)
        ,div_mod_sk_iters_per_region(div_mod_sk_iters_per_region)
        ,div_mod_iters_per_tile(div_mod_iters_per_tile)
        ,partials_workspace_bytes(partials_workspace_bytes)
        ,barrier_workspace_bytes(barrier_workspace_bytes)
        ,sk_workspace_size(sk_workspace_size)
        ,tile_workspace_size(tile_workspace_size)
        ,entries_per_block(entries_per_block)
         {}
  };

  struct TileIdxOffset {
    int problem_idx;
    int m; // row
    int n; // col
  };

  struct SharedStorage {
    // Sequence of problem IDs and starting tiles to compute
    cutlass::Array<ProblemInfo, kPrefetchTileCount> prefetched_problems;
  };


  //
  // dp_params
  //

  int32_t tiles_computed;
  int32_t iterations_per_block;
  int32_t block_load_start;
  ProblemInfo *problem_info_ptr;

  // shared
  SharedStorage &shared_storage;

  //
  // sk params
  //
  TileIdxOffset *tile_idx_offset_ptr;
  //skInfo *sk_runtime_ptr;
  skInfo sk_runtime_info;

  void *barrier_ptr;
  void *partials_ptr;

  bool is_sk;
  int block_iter_begin, block_iter_end, block_iters_remaining;
  int sk_tile_idx;

  // a runtime struct to hold sk-related info;
  struct TileWorkDesc
  {
    /// The linear tile index
    int tile_idx;

    /// The location of this tile (in threadblock-tile coordinates) in the output matrix
    cutlass::gemm::GemmCoord tiled_coord;

    // The first global-scoped MAC-iteration this threadblock will perform for this tile
    int iter_begin;

    // The starting index in the k-domain for MAC-iterations this threadblock will perform for this tile
    int k_begin;

    // The ending index (one-past) in the k-domain for MAC-iterations this threadblock will perform for this tile
    int k_end;

    /// The number of remaining MAC-iterations this threadblock will perform for this tile
    int k_iters_remaining;

    int problem_idx;

    // Whether this block will perform the first iteration of this tile
    CUTLASS_DEVICE
    bool tile_started() const
    {
      return (k_begin == 0);
    }

    // Whether this block will perform the last iteration of this tile
    CUTLASS_DEVICE
    bool tile_finished(int problem_size_k) const
    {
      return (k_end == problem_size_k);
    }
  };
  TileWorkDesc sk_tile_work{};


  //
  // heuristic
  //

  //
  static int const kMinItersPerSkBlock = 2;

  /// Height in CTAs of a grid rasterization cohort
  // static int const kCohortCtasM = 8;
  // /// Width in CTAs of a grid rasterization cohort
  // static int const kCohortCtasN = 4;
  // static int const kCtasPerCohort = kCohortCtasN * kCohortCtasM;


  //
  // sk-method
  //
  CUTLASS_DEVICE
  void get_iter_extents(
      int sk_block_idx,
      int &sk_iters_per_normal_block,
      int &block_iter_begin,
      int &block_iter_end) const
  {
    int block_idx_in_region = sk_block_idx;
    //int region_idx = 0;
    int sk_big_blocks_per_region = this->sk_runtime_info.sk_big_blocks_per_region;

    // only one region
    //assert(region_idx == 0);
    //assert(block_idx_in_region==sk_block_idx);
    //

    // block_iter_begin = (region_idx * sk_iters_per_region) + (block_idx_in_region * sk_iters_per_normal_block);
    block_iter_begin = block_idx_in_region * sk_iters_per_normal_block;

    // Adjust extents for the first "num_big_blocks" blocks that get one extra iteration
    int block_iters = sk_iters_per_normal_block;
    if (block_idx_in_region < sk_big_blocks_per_region) {
      // This is a +1 iteration block
      block_iter_begin += block_idx_in_region;
      block_iters++;
    } else {
      // This is a regular block
      block_iter_begin += sk_big_blocks_per_region;
    }
    block_iter_end = block_iter_begin + block_iters;
  }

  CUTLASS_DEVICE
  int get_sk_tile_idx(int iter, int iters_per_tile) const
  {
    //
    // it seems use div_mod_iters_per_tile is slower on RTX4090
    // but A100 is faster...
    //
    int tile_idx = this->sk_runtime_info.div_mod_iters_per_tile.div(iter);
    return tile_idx;


    //return iter/iters_per_tile;
  }

  CUTLASS_DEVICE
  void init_sk_tile_work(
      TileWorkDesc &tile_work,
      int tile_idx,
      int block_iter_begin,
      int block_iter_end)
  {
    // The linear tile index
    tile_work.tile_idx = tile_idx;

    // The first global-scoped MAC-iteration for this tile
    // int tile_iter_begin = tile_idx * params.block_mapping.iters_per_tile();
    int tile_iter_begin = tile_idx * this->sk_runtime_info.iters_per_tile;

    // The first global-scoped MAC-iteration this threadblock will perform for this tile
    tile_work.iter_begin = max(block_iter_begin, tile_iter_begin);

    // The first tile-scoped MAC-iteration this threadblock will perform for this tile
    int k_iter_begin = tile_work.iter_begin - tile_iter_begin;

    // The last (one past) tile-scoped MAC-iteration this threadblock will perform for this tile
    int k_iter_end = block_iter_end - tile_iter_begin;

    // The number of MAC-iterations this threadblock will perform for this tile
    tile_work.k_iters_remaining = k_iter_end - k_iter_begin;

    // The starting index in the k-domain for MAC-iterations this threadblock will perform for this tile
    //tile_work.k_begin = k_iter_begin * Mma::Shape::kK;
    tile_work.k_begin = k_iter_begin * this->sk_runtime_info.mma_shape_k;

    // The ending index (one-past) in the k-domain for MAC-iterations this threadblock will perform for this tile
    //tile_work.k_end = min(
    //    params.block_mapping.problem_size.k(),            // extent of k domain
    //    (k_iter_end * Mma::Shape::kK));                   // extent of the threadblock's global iteration assignment
    tile_work.k_end = min(
        this->sk_runtime_info.problem_size_k,            // extent of k domain
        (k_iter_end * this->sk_runtime_info.mma_shape_k)
    );

    // The location of this tile (in threadblock-tile coordinates) in the output matrix
    //tile_work.tiled_coord = params.block_mapping.get_tile_offset(tile_work.tile_idx);
    int m = this->tile_idx_offset_ptr[tile_idx].m;
    int n = this->tile_idx_offset_ptr[tile_idx].n;
    int problem_idx = this->tile_idx_offset_ptr[tile_idx].problem_idx;
    // cudaAssert(problem_idx==0);  
    tile_work.tiled_coord = cutlass::gemm::GemmCoord(m, n, 0);
    tile_work.problem_idx = problem_idx;
  }

  CUTLASS_DEVICE
  int get_first_block_idx(int tile_idx) const
  {
    // global-scope
    int iter = tile_idx * this->sk_runtime_info.iters_per_tile;

    // assume num_region = 1
    int region_idx = 0;
    int iter_in_region = 0;
    this->sk_runtime_info.div_mod_sk_iters_per_region(region_idx, iter_in_region, iter);
    // assert(region_idx == 0);
    // assert(iter_in_region == iter);
    int sk_big_blocks_per_region = this->sk_runtime_info.sk_big_blocks_per_region;

    // number of iterations in the region's big blocks
    int big_block_iters = (sk_big_blocks_per_region * this->sk_runtime_info.sk_iters_per_normal_block) + sk_big_blocks_per_region;   
    // number of iterations in the region's normal blocks
    int normal_block_iters = iter_in_region - big_block_iters; 

    int big_block_idx_in_region = this->sk_runtime_info.div_mod_sk_iters_per_big_block.div(iter_in_region);
    int normal_block_idx_in_region = sk_big_blocks_per_region + this->sk_runtime_info.div_mod_sk_iters_per_normal_block.div(normal_block_iters);


    int block_idx_in_region = (big_block_idx_in_region < sk_big_blocks_per_region) ?
        big_block_idx_in_region :
        normal_block_idx_in_region;

    //int owning_block_idx = (sk_blocks_per_region() * region_idx) + block_idx_in_region;
    int owning_block_idx = block_idx_in_region;

    // if ((blockIdx.x < 3) && threadIdx.x == 0)
    //   printf("[Get First SK BLOCK]: blockIdx.x: %d, iter: %d, iter_in_region: %d, big_block_iters: %d, normal_block_iters: %d, big_block_idx_in_region: %d, normal_block_idx_in_region: %d, block_idx_in_region: %d, owning_block_idx: %d\n", blockIdx.x, iter, iter_in_region, big_block_iters, normal_block_iters, big_block_idx_in_region, normal_block_idx_in_region, block_idx_in_region, owning_block_idx);

    return owning_block_idx;
  }

  //
  // Methods
  //
  CUTLASS_DEVICE
  GroupedProblemVisitor(
    Params &params_,
    SharedStorage &shared_storage_,
    int32_t block_idx
  ): Base(params_, block_idx/*tile_idx*/),
  tiles_computed(0),
  shared_storage(shared_storage_),
  //problem_info_ptr(reinterpret_cast<ProblemInfo const*>(params_.workspace)),
  tile_idx_offset_ptr(nullptr),
  barrier_ptr(nullptr),
  partials_ptr(nullptr),
  problem_info_ptr(nullptr)
  {
    uint8_t *ptr = reinterpret_cast<uint8_t*>(params_.workspace);

    //
    // sk related (can we move this to static schedule? and just prefetch?)
    //
    skInfo *sk_info_ptr = reinterpret_cast<skInfo*>(params_.workspace);
    skInfo sk_info = *sk_info_ptr;
    this->sk_runtime_info = sk_info;
    ptr += sk_info.sk_workspace_size;

    int dp_start_block_idx = sk_info.sk_blocks * sk_info.sk_waves;
    // int dp_start_tile_idx = sk_info.sk_tiles;

    // if (block_idx == 0 && threadIdx.x == 0) {
    //   printf("[SK]: dp_start_block_idx: %d, dp_start_tile_idx: %d, sk_iters: %d\n", dp_start_block_idx, dp_start_tile_idx, this->sk_runtime_ptr->sk_iters_per_normal_block);
    // }
    if (dp_start_block_idx > 0) {
      this->is_sk = true;

      // init sk global-scope block extent
      int block_iter_begin, block_iter_end, block_iters_remaining;
      get_iter_extents(
        block_idx,
        sk_info.sk_iters_per_normal_block,
        block_iter_begin,
        block_iter_end);
      block_iters_remaining = block_iter_end - block_iter_begin;
      this->block_iter_begin = block_iter_begin;
      this->block_iter_end = block_iter_end;
      this->block_iters_remaining = block_iters_remaining;

      // if (block_idx==107 && threadIdx.x==0) {
      //   printf("[Check-block-iters] block_iter_begin: %d, block_iter_end: %d, block_iters_remaining: %d\n", block_iter_begin, block_iter_end, block_iters_remaining);
      // }
      //this->sk_tile_work = TileWorkDesc{};
      this->sk_tile_idx = get_sk_tile_idx(block_iter_end - 1, sk_info.iters_per_tile);

    }
    else
      this->is_sk = false;

    // partials and barrier ptr
    this->partials_ptr = ptr;
    ptr+=sk_info.partials_workspace_bytes;

    this->barrier_ptr = ptr;
    ptr+=sk_info.barrier_workspace_bytes;

    // tile offset ptr
    this->tile_idx_offset_ptr = reinterpret_cast<TileIdxOffset *>(ptr);
    //ptr+=sk_info.sk_tiles * sizeof(TileIdxOffset);
    ptr+=sk_info.tile_workspace_size;

      // //sanity check
      // if (block_idx == 0 && threadIdx.x == 0) {
      //   for (int i = 0; i < sk_info.sk_tiles; i++) {
      //     printf("[Check-SK-Schedule] tile_idx_offset_ptr[%d] = (%d, %d, %d)\n", i, this->tile_idx_offset_ptr[i].problem_idx, this->tile_idx_offset_ptr[i].m, this->tile_idx_offset_ptr[i].n);
      //   }
      // }

    //if ((block_idx == 0  ) && threadIdx.x == 0) // error
    //if ((block_idx == 0 || block_idx == 1 ) && threadIdx.x == 0)
    ////  // __nanosleep(1000000);  sleep for 1ms also gets error
    //  printf("%d", block_idx);


    //
    // dp related
    //

    // iterations_per_block = (params_.tile_count - 1 + gridDim.x) / gridDim.x;
    // block_load_start = iterations_per_block * block_idx;

    this->iterations_per_block = sk_info.entries_per_block;
    this->block_load_start = sk_info.entries_per_block * block_idx;

    // advance to problem info ptr
    this->problem_info_ptr = reinterpret_cast<ProblemInfo *>(ptr);


    //if ((block_idx == 0 || block_idx == 1 || block_idx==127) && threadIdx.x == 0) {

    //  printf("[ProblemVisitor]: Bid: %d, tile_count: %d, tiles_computed: %d, tile_idx: %d, problem_tile_start: %d, problem_idx: %d, iterations_per_block: %d, block_load_start: %d, kPrefetchTileCount: %d, kThreadCount: %d\n", block_idx, params_.tile_count, tiles_computed, this->tile_idx, this->problem_tile_start, this->problem_idx, iterations_per_block, block_load_start, kPrefetchTileCount, kThreadCount);

    //  printf("\n");

    //}

    // Start prefetching the first set of tiles to compute
    prefetch_tiles();
  }


  CUTLASS_DEVICE
  bool next_tile() {
    if (this->is_sk) {
      // NOTE: advance() will turn off sk flag

      init_sk_tile_work(this->sk_tile_work, this->sk_tile_idx, this->block_iter_begin, this->block_iter_begin + this->block_iters_remaining);

      //int block_idx = blockIdx.x;
      //if ((block_idx == 0 || block_idx == 1 || block_idx==127) && threadIdx.x == 0)
      //  printf("[SK-Next]: Bid: %d, block_iter_begin: %d, block_iter_end: %d, block_iters_remaining: %d, sk_tile_idx: %d, tile_work.iter_begin: %d, tile_work.k_begin: %d, tile_work.k_iter_remaining: %d, tile_work.k_end: %d\n", block_idx, this->block_iter_begin, this->block_iter_end, this->block_iters_remaining, this->sk_tile_idx, this->sk_tile_work.iter_begin, this->sk_tile_work.k_begin, this->sk_tile_work.k_iters_remaining, this->sk_tile_work.k_end);

      //if ((blockIdx.x < 5 || blockIdx.x==127) && threadIdx.x == 0)

      return true;
    }


    //
    // start from here is dp logic:
    //

    if (this->tile_idx >= this->params.tile_count) {
      return false;
    }

    int32_t prefetch_idx = (tiles_computed % kPrefetchTileCount);
    if (prefetch_idx == 0) {
      // Ensure all previous stores to shared memory have been completed
      __syncthreads();
    }

    auto problem_info = shared_storage.prefetched_problems[prefetch_idx];
    ++tiles_computed;

    if ((tiles_computed % kPrefetchTileCount) == 0) {
      // Begin prefetching next set of tiles. Synchronize first to ensure that
      // we don't overwrite the current buffer while someone else is using it.
      __syncthreads();
      prefetch_tiles();
    }

    this->problem_idx = problem_info.problem_idx;
    this->problem_tile_start = problem_info.problem_start;

    // if ((blockIdx.x == 0) && threadIdx.x == 0) {
    //   printf("[DP-Next] tiled_idx: %d, prefetch_idx: %d, tiles_computed: %d, problem_idx: %d, problem_tile_start: %d\n", this->tile_idx, prefetch_idx, tiles_computed, this->problem_idx, this->problem_tile_start);
    //   ;
    // }

    return true;
  }


  //
  //
  // static methods
  //
  //

  // Pad the given allocation size up to the nearest cache line
  static size_t cacheline_align_up(size_t size)
  {
    static const int CACHELINE_SIZE = 128;
    return (size + CACHELINE_SIZE - 1) / CACHELINE_SIZE * CACHELINE_SIZE;
  }

  /// Get the workspace size needed for barrier
  static size_t get_barrier_workspace_size(skInfo& sk_info, size_t kWorkspaceBytesPerBlock)
  {
    // For atomic reduction, each SK-block needs a synchronization flag.  For parallel reduction,
    // each reduction block needs its own synchronization flag.
    // int sk_blocks = block_mapping.sk_regions() * block_mapping.sk_blocks_per_region();
    // int num_flags = fast_max(sk_blocks, block_mapping.reduction_blocks);

    int sk_blocks = sk_info.sk_blocks;
    int num_flags = sk_blocks;

    return cacheline_align_up(sizeof(typename Barrier::T) * num_flags);
  }

  /// Get the workspace size needed for intermediate partial sums
  static size_t get_partials_workspace_size(skInfo& sk_info, size_t kWorkspaceBytesPerBlock)
  {
    // int sk_blocks = block_mapping.sk_regions() * block_mapping.sk_blocks_per_region();
    int sk_blocks = sk_info.sk_blocks;
    return cacheline_align_up(kWorkspaceBytesPerBlock * sk_blocks);
  }

  static size_t get_workspace_size_streamk(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                                   int32_t problem_count,
                                   int32_t block_count,
                                   size_t kWorkspaceBytesPerBlock,
                                   //
                                  int num_sms,
                                  int sm_occupancy,
                                  int mma_shape_k,
                                  GemmCoord thread_block_shape
                                  //
                                   ) {

    auto sk_info = plan_sk(host_problem_sizes_ptr,
                      problem_count,
                      num_sms,
                      sm_occupancy,
                      thread_block_shape,
                      block_count,
                      mma_shape_k,
                      false);

    // int32_t total_tiles = Base::group_tile_count(host_problem_sizes_ptr, problem_count);
    // int32_t entries_per_block = ((total_tiles - 1 + block_count) / block_count);
    int32_t entries_per_block = static_cast<int32_t>(sk_info.entries_per_block);
    int sk_tiles = sk_info.sk_tiles;

    size_t barrier_partial_workspace_size = get_barrier_workspace_size(sk_info, kWorkspaceBytesPerBlock) + get_partials_workspace_size(sk_info, kWorkspaceBytesPerBlock);

    size_t sk_workspace_size = cacheline_align_up(sizeof(skInfo));
    size_t tile_workspace_size = cacheline_align_up(sizeof(TileIdxOffset) * sk_tiles);
    size_t problem_workspace_size= cacheline_align_up(sizeof(ProblemInfo) * entries_per_block * block_count);

    //return sizeof(skInfo) + barrier_partial_workspace_size  + sizeof(TileIdxOffset) * sk_tiles + sizeof(ProblemInfo) * entries_per_block * block_count;
    return sk_workspace_size + barrier_partial_workspace_size  + tile_workspace_size + problem_workspace_size;
  }

  static void host_precompute_streamk(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                              int32_t problem_count,
                              int32_t block_count,
                              size_t kWorkspaceBytesPerBlock,
                              //
                              int num_sms,
                              int sm_occupancy,
                              int mma_shape_k,
                              GemmCoord thread_block_shape,
                              //
                              void* host_workspace_ptr) {
    std::cout << "[Precompute] " << std::endl;

    uint8_t *ptr = reinterpret_cast<uint8_t*>(host_workspace_ptr);

    auto sk_info = plan_sk(host_problem_sizes_ptr,
                      problem_count,
                      num_sms,
                      sm_occupancy,
                      thread_block_shape,
                      block_count,
                      mma_shape_k,
                      true);

    size_t partials_workspace_bytes = get_partials_workspace_size(sk_info, kWorkspaceBytesPerBlock);
    size_t barrier_workspace_bytes = get_barrier_workspace_size(sk_info, kWorkspaceBytesPerBlock);

    sk_info.partials_workspace_bytes = partials_workspace_bytes;
    sk_info.barrier_workspace_bytes = barrier_workspace_bytes;

    size_t sk_workspace_size = cacheline_align_up(sizeof(skInfo));
    size_t tile_workspace_size = cacheline_align_up(sizeof(TileIdxOffset) * sk_info.sk_tiles);
    sk_info.sk_workspace_size = sk_workspace_size;
    sk_info.tile_workspace_size = tile_workspace_size;


    //
    // sk
    //
    {
      skInfo *sk_info_ptr = reinterpret_cast<skInfo*>(ptr);
      *sk_info_ptr = sk_info;
      //ptr += sizeof(skInfo);
      ptr += sk_workspace_size;
    }

    // partial-barrier
    {
      //uint8_t *partials_workspace = ptr;  // unused
      ptr += partials_workspace_bytes;

      //void* barrier_workspace = ptr;
      //memset(
      //  barrier_workspace,
      //  0,
      //  barrier_workspace_bytes);
      //int* iptr = (int*)ptr; // We tell the compiler that the values at the end of the pointer should be interpreted as integers
      //for(int i = 0; i < barrier_workspace_bytes; i++) {
      //  std::cout << i << " ";
      //  ASSERT(iptr[i] == 0);
      //}
      ptr += barrier_workspace_bytes;
    }


    //
    // deal with tile offset
    //
    {
      TileIdxOffset* host_tile_off_ptr = reinterpret_cast<TileIdxOffset*>(ptr);

      int sk_tiles = sk_info.sk_tiles;
      int tile_idx = 0;
      int start_tile = 0;
      for (int p_idx = 0; p_idx < problem_count; ++p_idx) {
        auto problem = host_problem_sizes_ptr[p_idx];
        Base::possibly_transpose_problem(problem);
        auto grid = Base::grid_shape(problem);
        int tiles = Base::tile_count(grid);

        // XXX cannot use ColumnMajor for now!!!
        static_assert(!ProblemSizeHelper::kTransposed);
        int grid_shape_base = grid.n();
        // printf("[DEBUG]: p_idx=%d, problem has tiles=%d, grid_shape_m=%d grid_shape_n=%d, transpose=%d\n", p_idx, tiles, grid.m(), grid.n() , ProblemSizeHelper::kTransposed);

        // ASSERT(p_idx==0);

        for (int i = 0; i < tiles; ++i, ++tile_idx) {

          if (tile_idx >= sk_tiles)
            break;

          int m = i/grid_shape_base;
          int n = i%grid_shape_base;

          TileIdxOffset tile_idx_offset;
          tile_idx_offset.problem_idx = p_idx;
          tile_idx_offset.m = m;
          tile_idx_offset.n = n;
          host_tile_off_ptr[tile_idx] = tile_idx_offset;
        }
        start_tile += tiles;

        if (tile_idx >= sk_tiles)
          break;
      }

      //ptr += sizeof(TileIdxOffset) * sk_tiles;
      ptr+=tile_workspace_size;
    }


    //
    // advance to problem info ptr
    //
    ProblemInfo* host_problem_info_ptr = reinterpret_cast<ProblemInfo*>(ptr);

    //
    // based on sk info, schedule the rest dp blocks
    //

    //// NOTE assign all to dp
    // int32_t total_tiles = Base::group_tile_count(host_problem_sizes_ptr, problem_count);
    // int32_t entries_per_block = (total_tiles - 1 + block_count) / block_count;
    //
    // NOTE assign the rest to dp
    int32_t entries_per_block = static_cast<int32_t>(sk_info.entries_per_block);

    int tile = 0;
    int start_tile = 0;
    for (int p_idx = 0; p_idx < problem_count; ++p_idx) {
      auto problem = host_problem_sizes_ptr[p_idx];
      Base::possibly_transpose_problem(problem);
      auto grid = Base::grid_shape(problem);
      int tiles = Base::tile_count(grid);
      ProblemInfo problem_info(p_idx, start_tile);
      for (int i = 0; i < tiles; ++i, ++tile) {

        //
        // for [0 - entries_per_block], it's the work belongs to block 0
        //
        // for problems 0, if it has 128 tiles (assume 128 sms), each sm takes one tile,
        // so its tiles are distributed across sms
        //

        if (tile < sk_info.sk_tiles) {
          ; // sk
        } else {
          int dp_work_assignment = tile - sk_info.sk_tiles; // start from 0
          host_problem_info_ptr[(entries_per_block * (dp_work_assignment % block_count)) + (dp_work_assignment / block_count)] = problem_info;
        }

        // host_problem_info_ptr[(entries_per_block * (tile % block_count)) + (tile / block_count)] = problem_info;
      }
      start_tile += tiles;
    }

    std::cout << "[Precompute Done]" << std::endl;
  }








  static size_t get_workspace_size(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                                   int32_t problem_count,
                                   int32_t block_count) { assert(false); return 0; }
  static void host_precompute(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                              int32_t problem_count,
                              int32_t block_count,
                              void* host_workspace_ptr) { assert(false);}

  // Compute sk_blocks to dispatch for a given number of sk_tiles
  static void get_sk_blocks(
    int &sk_blocks,     /// [out]
    int &savings_iters, /// [out]
    int sk_tiles,
    int iters_per_tile,
    int avail_sms,
    int max_sk_occupancy,
    bool allow_partial_wave,
    bool verbose)
  {
    savings_iters = INT_MIN;
    sk_blocks = 0;

    if (sk_tiles == 0) {
      return;
    }

    int sk_iters = sk_tiles * iters_per_tile;

    int dp_equiv_waves = (sk_tiles + avail_sms - 1) / avail_sms;
    int dp_equiv_iters = iters_per_tile * dp_equiv_waves;

    int min_sk_blocks = (allow_partial_wave) ? fast_min(avail_sms, sk_tiles + 1) : avail_sms;
    int max_sk_blocks = fast_min(avail_sms * max_sk_occupancy, sk_iters / kMinItersPerSkBlock);

    if (verbose)
      std::cout << "sk_tiles: " << sk_tiles << ", sk_iters: " << sk_iters << ", max_sk_occupancy: " << max_sk_occupancy << ", dp_equiv_waves: " << dp_equiv_waves << ", dp_equiv_iters: " << dp_equiv_iters << ", min_sk_blocks: " << min_sk_blocks << ", max_sk_blocks: " << max_sk_blocks << std::endl;

    for (int trial_sk_blocks = min_sk_blocks; trial_sk_blocks <= max_sk_blocks; ++trial_sk_blocks)
    {
      int sk_waves = (trial_sk_blocks + avail_sms - 1) / avail_sms;
      int max_sk_iters_per_block = (sk_iters + trial_sk_blocks - 1) / trial_sk_blocks;
      int sk_iter_equiv = max_sk_iters_per_block * sk_waves;

      int num_peers = ((trial_sk_blocks + sk_tiles - 1) / sk_tiles) + 1;        // add one for alignment skew

      float iter_cost = 0.02f * float(num_peers) * float(sk_iter_equiv);

      if (trial_sk_blocks % sk_tiles == 0)
      {
        // aligned
        num_peers = (trial_sk_blocks / sk_tiles);

        iter_cost = 0.0f;
      }

      float peer_cost = 2.0f * float(num_peers);

      float base_cost = 2.0f * float(sk_waves);

      int fixup_iter_equiv = int(base_cost + iter_cost + peer_cost);

      int trial_savings_iters = dp_equiv_iters - sk_iter_equiv - fixup_iter_equiv;

      if (trial_savings_iters >= savings_iters) {
          savings_iters = trial_savings_iters;
          sk_blocks = trial_sk_blocks;
      }
    }
  }

  static skInfo plan_sk(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                              int32_t problem_count,
                              int num_sms,
                              int sm_occupancy,
                              GemmCoord thread_block_shape,
                              int32_t block_count,
                              int mma_shape_k,
                              bool verbose
                              ) {
    // total tiles for all gemm
    int32_t total_tiles = Base::group_tile_count(host_problem_sizes_ptr, problem_count);

    //
    // Determine dispatch composition of DP-tiles and SK-blocks
    //
    auto first_problem = host_problem_sizes_ptr[0];
    int iters_per_tile = (first_problem.k() + thread_block_shape.k() - 1) / thread_block_shape.k();
    int waves = (total_tiles + num_sms - 1) / num_sms;
    float dp_efficiency = float(total_tiles) / float(waves * num_sms);

    if (verbose)
      std::cout << "total_tiles: " << total_tiles << ", iters_per_tile: " << iters_per_tile << ", waves: " << waves << ", dp_efficiency: " << dp_efficiency << std::endl;

    // Start with a DP-only configuration
    int dp_tiles = total_tiles;     // Number of data-parallel tiles
    int sk_blocks = 0;

    //
    // get_blocks
    //
    bool dp_only = false;
    int full_waves = total_tiles / num_sms;
    int full_waves_tiles = full_waves * num_sms;
    int partial_wave_tiles = total_tiles - full_waves_tiles;
    int score = -1;
    if (partial_wave_tiles == 0) {
      // Perfect quantization
      // ASSERT(false);
      dp_only=true;
    }
    if (full_waves < sm_occupancy) {
      // TODO cornor case: We're less than full GPU occupancy
      ASSERT(false);
    }
    if ((sm_occupancy > 1 ) && (full_waves % sm_occupancy == sm_occupancy - 1)) {
      //
      // this is common for A100
      //
      //int max_sk_occupancy = 1;
      //dp_tiles = full_waves_tiles;

      //get_sk_blocks(
      //  sk_blocks,
      //  score,
      //  partial_wave_tiles,
      //  iters_per_tile,
      //  num_sms,
      //  max_sk_occupancy,
      //  true,           // we can run with less than a full wave of SK-blocks
      //  verbose
      //  );                 

      std::cout << "[warning] just skip for now... (should be affecting A100)" << std::endl;

      // if (score >= 0)
      //     return;
      //ASSERT(false);
    }

    if (score<0) {
      dp_tiles = full_waves_tiles - num_sms;  // this gives 1 wave to sk
      int max_sk_occupancy = sm_occupancy - ((full_waves - 1) % sm_occupancy);
      get_sk_blocks(
        sk_blocks,
        score,
        partial_wave_tiles + num_sms,
        iters_per_tile,
        num_sms,
        max_sk_occupancy,
        false, // we cannot run with less than a full wave of SK-blocks
        verbose);                 

      if (score < 0) {
        // not profitable; apply a dp setting
        // ASSERT(false);
        dp_only = true;
      }
    }

    //
    // post-process sk region
    //
    int sk_tiles = total_tiles - dp_tiles;
    int sk_regions = 1;
    int sk_iters_per_normal_block;
    int sk_big_blocks_per_region;
    // int reduction_blocks = 0;
    // bool remap_block_indices = false;
    int sk_waves = -1;
    int sk_iters_per_region;
    //int sk_blocks_per_region ;
    ASSERT(sk_blocks > 0);
    {
      sk_waves = (sk_blocks + num_sms - 1) / num_sms;
      int sk_iters = sk_tiles * iters_per_tile;
      sk_blocks = fast_min(sk_blocks, sk_iters);
      sk_iters_per_normal_block = sk_iters / sk_blocks;
      int extra_sk_iters = sk_iters - (sk_iters_per_normal_block * sk_blocks);
      int sk_big_blocks = extra_sk_iters;
      // if (sk_big_blocks>0) {
      //   ASSERT(false && "a problem with sk big blocks");
      // }

      if ((sk_blocks > sk_tiles) && (sk_blocks % sk_tiles == 0)) {
        // // Split-K decomposition
        // sk_regions = sk_tiles;
        std::cout << "sk_blocks: " << sk_blocks << ", sk_tiles: " << sk_tiles << std::endl;
        ASSERT(false);
      }

      //sk_blocks_per_region = sk_blocks / sk_regions;
      sk_big_blocks_per_region = sk_big_blocks / sk_regions;
      sk_iters_per_region = sk_iters / sk_regions;
    }
    FastDivmod div_mod_sk_iters_per_normal_block(sk_iters_per_normal_block);
    FastDivmod div_mod_sk_iters_per_big_block(sk_iters_per_normal_block+1);
    FastDivmod div_mod_sk_iters_per_region(sk_iters_per_region);
    FastDivmod div_mod_iters_per_tile(iters_per_tile);


    //
    // Compute DP blocks
    // TODO: dont think we need cohort raster in grouped GEMM
    //

    int dp_blocks = dp_tiles;
    int entries_per_block = dp_tiles/block_count;
    // GemmCoord tiled_shape(
    //   (first_problem.m() + thread_block_shape.m() - 1) / thread_block_shape.m(),
    //   (first_problem.n() + thread_block_shape.n() - 1) / thread_block_shape.n(),
    //   1);
    // cutlass::gemm::GemmCoord tiled_cohort_shape(
    //     (tiled_shape.m() + kCohortCtasM - 1) / kCohortCtasM,
    //     (tiled_shape.n() + kCohortCtasN - 1) / kCohortCtasN,
    //     tiled_shape.k());
    // int cohort_blocks = (tiled_cohort_shape.m() * tiled_cohort_shape.n()) * kCtasPerCohort;
    // float cohort_efficiency = float(dp_blocks) / float(cohort_blocks);

    if (dp_only) {
      // overwrite all sk setting
      sk_regions = 0;
      dp_blocks = total_tiles;
      dp_tiles = total_tiles;     
      sk_blocks = 0;
      sk_tiles = 0;
      sk_waves = 0;
      sk_iters_per_normal_block = 0;
      entries_per_block = (total_tiles - 1 + block_count) / block_count;
    }

    //
    //ASSERT(dp_tiles % block_count == 0);  // <- perfect quantization!!!
    if (dp_tiles % block_count != 0) {
      ASSERT(dp_only);
      std::cout << "[Reset to dp]: dp_only: " << dp_only << ", dp_tiles: " << dp_tiles << ", block_count: " << block_count << ", dp_efficiency: " << dp_efficiency << ", score: " << score << std::endl;
    }

    skInfo sk_info = skInfo(
         sk_regions,
         dp_blocks,
         dp_tiles,
         sk_blocks,
         //sk_blocks_per_region,
         sk_tiles,
         sk_waves,
         sk_iters_per_normal_block,
        sk_big_blocks_per_region,
         iters_per_tile,
         first_problem.k(),  // all problem size k must be the same for MoE
         mma_shape_k,
         thread_block_shape,
         //
        div_mod_sk_iters_per_normal_block,
         div_mod_sk_iters_per_big_block,
         div_mod_sk_iters_per_region,
         div_mod_iters_per_tile,
         //
         0,
         0,
         0,
         0,
         //
         entries_per_block
    );
    if (verbose)
      std::cout << "dp_only: " << dp_only << ", sk regions: " << sk_info.sk_regions << ", dp blocks: " << sk_info.dp_blocks << ", dp tiles: " << sk_info.dp_tiles << ", sk blocks: " << sk_info.sk_blocks << ", sk tiles: " << sk_info.sk_tiles << ", sk_waves: " << sk_info.sk_waves << ", sk_iters_per_normal_block: " << sk_big_blocks_per_region << ", sk_big_blocks_per_region: " << sk_big_blocks_per_region <<  std::endl;
    return sk_info;
  }





private:
  CUTLASS_DEVICE
  void prefetch_tiles() {

    //
    // this is dp logic!!!
    //

    CUTLASS_PRAGMA_UNROLL
    for (int32_t i = 0; i < kPrefetchTileCount; i += kThreadCount) {
      int32_t offset = threadIdx.x + i;
      if (offset < kPrefetchTileCount && (tiles_computed + offset < iterations_per_block)) {
        shared_storage.prefetched_problems[offset] = problem_info_ptr[block_load_start + tiles_computed + offset];
      }
    }
  }

public:
  CUTLASS_DEVICE
  void advance(int32_t grid_size) {
    if (this->is_sk) {

      this->block_iters_remaining -= this->sk_tile_work.k_iters_remaining;
      if (this->block_iters_remaining == 0) {

        // switch to dp
        this->tile_idx += this->sk_runtime_info.sk_tiles;
        this->is_sk = false;

        // if ((blockIdx.x == 0 ) && threadIdx.x == 0)
        //   printf("\n");
      } else {

        // continue to next SK-work
        this->sk_tile_idx-=1; // SK blocks consume their tiles in backwards order

        // Continue to next tile; 
        __syncthreads();
      }

    } else {
      this->tile_idx += grid_size;
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
