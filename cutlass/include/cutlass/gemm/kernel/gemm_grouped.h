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
    \brief Problem visitor for grouped GEMMs
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/complex.h"
#include "cutlass/semaphore.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/trace.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/gemm/kernel/gemm_grouped_problem_visitor.h"

#include "cutlass/barrier.h"
#include "cutlass/block_striped.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
#define cudaAssert(condition) \
  if (!(condition)){ printf("Assertion %s failed!\n", #condition); asm("trap;"); }

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma_,                           ///! Threadblock-scoped matrix multiply-accumulate
  typename Epilogue_,                      ///! Epilogue
  typename ThreadblockSwizzle_,            ///! Threadblock swizzling function
  GroupScheduleMode GroupScheduleMode_,    ///! Type of scheduling to perform
  bool Transposed = false
>
struct GemmGrouped {
public:

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static GroupScheduleMode const kGroupScheduleMode = GroupScheduleMode_;
  static bool const kTransposed = Transposed;

  // Optional transpose
  using MapArguments = kernel::detail::MapArguments<
    typename Mma::IteratorA::Element,
    typename Mma::IteratorA::Layout,
    Mma::kTransformA,
    Mma::IteratorA::AccessType::kElements,
    typename Mma::IteratorB::Element,
    typename Mma::IteratorB::Layout,
    Mma::kTransformB,
    Mma::IteratorB::AccessType::kElements,
    typename Mma::LayoutC,
    kTransposed
  >;

  // Public-facing type definitions related to operand element type, layout, and complex conjugate
  // operation. Must interact with the 'kTransposed' notion.
  using ElementA = typename MapArguments::ElementA;
  using LayoutA = typename MapArguments::LayoutA;
  using ElementB = typename MapArguments::ElementB;
  using LayoutB = typename MapArguments::LayoutB;
  using ElementC = typename Epilogue::OutputTileIterator::Element;
  using LayoutC = typename MapArguments::LayoutC;

  static ComplexTransform const kTransformA = MapArguments::kTransformA;
  static ComplexTransform const kTransformB = MapArguments::kTransformB;

  // Type definitions about the mainloop.
  using Operator = typename Mma::Operator;
  using OperatorClass = typename Mma::Operator::OperatorClass;
  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename Mma::Operator::Shape;
  using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
  using ArchTag = typename Mma::ArchTag;

  static int const kStages = Mma::kStages;
  static int const kAlignmentA = MapArguments::kAlignmentA;
  static int const kAlignmentB = MapArguments::kAlignmentB;
  static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  using ProblemVisitor = GemmGroupedProblemVisitor<
                            ThreadblockShape,
                            kGroupScheduleMode,
                            kThreadCount,
                            kThreadCount,
                            kTransposed>;

  using AccumulatorTile = typename Mma::FragmentC;
  static size_t const kWorkspaceBytesPerBlock =
    __NV_STD_MAX(
      kThreadCount * sizeof(AccumulatorTile),
      Epilogue::kWorkspaceBytesPerBlock);

  //
  // Structures
  //

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmCoord *problem_sizes{nullptr};
    int problem_count{0};
    int threadblock_count{0};

    typename EpilogueOutputOp::Params output_op{};

    ElementA ** ptr_A{nullptr};
    ElementB ** ptr_B{nullptr};
    ElementC ** ptr_C{nullptr};
    ElementC ** ptr_D{nullptr};

    typename LayoutA::Stride::LongIndex *lda{nullptr};
    typename LayoutB::Stride::LongIndex *ldb{nullptr};
    typename LayoutC::Stride::LongIndex *ldc{nullptr};
    typename LayoutC::Stride::LongIndex *ldd{nullptr};

    // Only used by device-level operator
    GemmCoord *host_problem_sizes{nullptr};


    //
    // Methods
    //

    /// Default ctor
    Arguments() = default;

    /// Ctor
    CUTLASS_HOST_DEVICE
    Arguments(    
      GemmCoord *problem_sizes,
      int problem_count,
      int threadblock_count,
      typename EpilogueOutputOp::Params output_op,
      ElementA ** ptr_A,
      ElementB ** ptr_B,
      ElementC ** ptr_C,
      ElementC ** ptr_D,
      typename LayoutA::Stride::LongIndex *lda,
      typename LayoutB::Stride::LongIndex *ldb,
      typename LayoutC::Stride::LongIndex *ldc,
      typename LayoutC::Stride::LongIndex *ldd,
      GemmCoord *host_problem_sizes=nullptr
    ): 
      problem_sizes(problem_sizes),
      problem_count(problem_count),
      threadblock_count(threadblock_count),
      output_op(output_op),
      ptr_A(ptr_A),
      ptr_B(ptr_B),
      ptr_C(ptr_C),
      ptr_D(ptr_D),
      lda(lda),
      ldb(ldb),
      ldc(ldc),
      ldd(ldd),
      host_problem_sizes(host_problem_sizes)
    {

    }
  };

  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure
  struct Params {

    typename ProblemVisitor::Params problem_visitor{};
    int threadblock_count{0};

    typename EpilogueOutputOp::Params output_op{};

    ElementA ** ptr_A{nullptr};
    ElementB ** ptr_B{nullptr};
    ElementC ** ptr_C{nullptr};
    ElementC ** ptr_D{nullptr};

    typename LayoutA::Stride::LongIndex *lda{nullptr};
    typename LayoutB::Stride::LongIndex *ldb{nullptr};
    typename LayoutC::Stride::LongIndex *ldc{nullptr};
    typename LayoutC::Stride::LongIndex *ldd{nullptr};

    //
    // Methods
    //

    Params() = default;

    CUTLASS_HOST_DEVICE
    Params(Arguments const &args,
          void *workspace = nullptr,
          int tile_count = 0):
      problem_visitor(args.problem_sizes, args.problem_count, workspace, tile_count),
      threadblock_count(args.threadblock_count),
      output_op(args.output_op),
      ptr_A(args.ptr_A),
      ptr_B(args.ptr_B),
      ptr_C(args.ptr_C),
      ptr_D(args.ptr_D),
      lda(args.lda),
      ldb(args.ldb),
      ldc(args.ldc),
      ldd(args.ldd)
    { 
      printf("[Init Params]: threadblock_count = %d, tile_count = %d, problem_count = %d\n", threadblock_count, tile_count, args.problem_count);
    }

    CUTLASS_HOST_DEVICE
    void update(
      Arguments const &args,
      void *workspace = nullptr,
      int tile_count = 0) {

      problem_visitor = typename ProblemVisitor::Params(args.problem_sizes, args.problem_count,
                                                        workspace, tile_count);
      threadblock_count = args.threadblock_count;
      output_op = args.output_op;
      ptr_A = args.ptr_A;
      ptr_B = args.ptr_B;
      ptr_C = args.ptr_C;
      ptr_D = args.ptr_D;
      lda = args.lda;
      ldb = args.ldb;
      ldc = args.ldc;
      ldd = args.ldd;
    }
  };

  /// Shared memory storage structure
  struct SharedStorage {
    union {
      typename Mma::SharedStorage main_loop;
      typename Epilogue::SharedStorage epilogue;
    } kernel;

    // ProblemVisitor shared storage can't be overlapped with others
    typename ProblemVisitor::SharedStorage problem_visitor;
  };

public:

  //
  // Methods
  //

  CUTLASS_DEVICE
  GemmGrouped() { } 

  /// Determines whether kernel satisfies alignment
  static Status can_implement(cutlass::gemm::GemmCoord const & problem_size) {
    return Status::kSuccess;
  }

  static Status can_implement(Arguments const &args) {
    return Status::kSuccess;
  }
 
  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params &params, SharedStorage &shared_storage) {

    //
    // These types shadow the type-level definitions and support the ability to implement
    // a 'transposed' GEMM that computes the transposed problems.
    //
    using ElementA = typename Mma::IteratorA::Element;
    using LayoutA = typename Mma::IteratorA::Layout;
    using ElementB = typename Mma::IteratorB::Element;
    using LayoutB = typename Mma::IteratorB::Layout;
    using ElementC = typename Epilogue::OutputTileIterator::Element;
    using LayoutC = typename Epilogue::OutputTileIterator::Layout;

    //
    // Problem visitor.
    //
    ProblemVisitor problem_visitor(
      params.problem_visitor,
      shared_storage.problem_visitor,
      blockIdx.x);

    // Outer 'persistent' loop to iterate over tiles
    while (problem_visitor.next_tile()) {

      GemmCoord problem_size  = problem_visitor.problem_size();
      int32_t problem_idx     = problem_visitor.problem_index();
      int32_t threadblock_idx = int32_t(problem_visitor.threadblock_idx());

      GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

      cutlass::gemm::GemmCoord threadblock_offset(
        int(threadblock_idx / grid_shape.n()) * Mma::Shape::kM,
        int(threadblock_idx % grid_shape.n()) * Mma::Shape::kN,
        0);

      // Load element pointers. Exchange pointers and strides if working on the transpose
      ElementA *ptr_A = reinterpret_cast<ElementA *>((kTransposed ? params.ptr_B[problem_idx] : params.ptr_A[problem_idx]));
      typename LayoutA::LongIndex ldm_A = (kTransposed ? params.ldb[problem_idx] : params.lda[problem_idx]);

      ElementB *ptr_B = reinterpret_cast<ElementB *>((kTransposed ? params.ptr_A[problem_idx] : params.ptr_B[problem_idx]));
      typename LayoutB::LongIndex ldm_B = (kTransposed ? params.lda[problem_idx] : params.ldb[problem_idx]);

      // Compute initial location in logical coordinates
      cutlass::MatrixCoord tb_offset_A{
        threadblock_offset.m(),
        0,
      };

      cutlass::MatrixCoord tb_offset_B{
        0,
        threadblock_offset.n()
      };

      // Compute position within threadblock
      int thread_idx = threadIdx.x;

      // Construct iterators to A and B operands
      typename Mma::IteratorA iterator_A(
        LayoutA(ldm_A),
        ptr_A,
        {problem_size.m(), problem_size.k()},
        thread_idx,
        tb_offset_A);

      typename Mma::IteratorB iterator_B(
        LayoutB(ldm_B),
        ptr_B,
        {problem_size.k(), problem_size.n()},
        thread_idx,
        tb_offset_B);

      typename Mma::FragmentC accumulators;

      accumulators.clear();
      
      // Broadcast the warp_id computed by lane 0 to ensure dependent code
      // is compiled as warp-uniform.
      int warp_idx = canonical_warp_idx_sync();

      int lane_idx = threadIdx.x % 32;

      //
      // Matrix multiply phase
      //

      // Construct thread-scoped matrix multiply
      Mma mma(shared_storage.kernel.main_loop, thread_idx, warp_idx, lane_idx);

      // Compute threadblock-scoped matrix multiply-add
      int gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Wait for all threads to finish their epilogue phases from the previous tile.
      __syncthreads();

      // Compute threadblock-scoped matrix multiply-add
      mma(
        gemm_k_iterations, 
        accumulators, 
        iterator_A, 
        iterator_B, 
        accumulators);

      //
      // Epilogue
      //

      EpilogueOutputOp output_op(params.output_op);

      ElementC *ptr_C = params.ptr_C[problem_idx];
      ElementC *ptr_D = params.ptr_D[problem_idx];

      LayoutC layout_C(params.ldc[problem_idx]);
      LayoutC layout_D(params.ldd[problem_idx]);

      typename Epilogue::OutputTileIterator::Params params_C(layout_C);
      typename Epilogue::OutputTileIterator::Params params_D(layout_D);

      // Tile iterator loading from source tensor.
      typename Epilogue::OutputTileIterator iterator_C(
        params_C,
        ptr_C,
        problem_size.mn(),
        thread_idx,
        threadblock_offset.mn()
      );

      // Tile iterator writing to destination tensor.
      typename Epilogue::OutputTileIterator iterator_D(
        params_D,
        ptr_D,
        problem_size.mn(),
        thread_idx,
        threadblock_offset.mn()
      );

      Epilogue epilogue(
        shared_storage.kernel.epilogue, 
        thread_idx, 
        warp_idx, 
        lane_idx);

      // Execute the epilogue operator to update the destination tensor.
      epilogue(
        output_op, 
        iterator_D, 
        accumulators, 
        iterator_C); 

      // Next tile
      problem_visitor.advance(gridDim.x);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma_,                           ///! Threadblock-scoped matrix multiply-accumulate
  typename Epilogue_,                      ///! Epilogue
  typename ThreadblockSwizzle_,            ///! Threadblock swizzling function
  //GroupScheduleMode GroupScheduleMode_,    ///! Type of scheduling to perform
  bool Transposed
>
struct GemmGrouped<Mma_,
                  Epilogue_, 
                  ThreadblockSwizzle_, 
                  GroupScheduleMode::mixedStreamK,  // partial specification
                  Transposed> {
public:

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static GroupScheduleMode const kGroupScheduleMode = GroupScheduleMode::mixedStreamK;
  static bool const kTransposed = Transposed;

  // Optional transpose
  using MapArguments = kernel::detail::MapArguments<
    typename Mma::IteratorA::Element,
    typename Mma::IteratorA::Layout,
    Mma::kTransformA,
    Mma::IteratorA::AccessType::kElements,
    typename Mma::IteratorB::Element,
    typename Mma::IteratorB::Layout,
    Mma::kTransformB,
    Mma::IteratorB::AccessType::kElements,
    typename Mma::LayoutC,
    kTransposed
  >;

  // Public-facing type definitions related to operand element type, layout, and complex conjugate
  // operation. Must interact with the 'kTransposed' notion.
  using ElementA = typename MapArguments::ElementA;
  using LayoutA = typename MapArguments::LayoutA;
  using ElementB = typename MapArguments::ElementB;
  using LayoutB = typename MapArguments::LayoutB;
  using ElementC = typename Epilogue::OutputTileIterator::Element;
  using LayoutC = typename MapArguments::LayoutC;

  static ComplexTransform const kTransformA = MapArguments::kTransformA;
  static ComplexTransform const kTransformB = MapArguments::kTransformB;

  // Type definitions about the mainloop.
  using Operator = typename Mma::Operator;
  using OperatorClass = typename Mma::Operator::OperatorClass;
  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename Mma::Operator::Shape;
  using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
  using ArchTag = typename Mma::ArchTag;

  static int const kStages = Mma::kStages;
  static int const kAlignmentA = MapArguments::kAlignmentA;
  static int const kAlignmentB = MapArguments::kAlignmentB;
  static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  using ProblemVisitor = GemmGroupedProblemVisitor<
                            ThreadblockShape,
                            kGroupScheduleMode,
                            kThreadCount,
                            kThreadCount,
                            kTransposed>;

  using AccumulatorTile = typename Mma::FragmentC;
  static size_t const kWorkspaceBytesPerBlock =
    __NV_STD_MAX(
      kThreadCount * sizeof(AccumulatorTile),
      Epilogue::kWorkspaceBytesPerBlock);

  using BlockStripedReduceT = BlockStripedReduce<kThreadCount, AccumulatorTile>;

  //
  // Structures
  //

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmCoord *problem_sizes{nullptr};
    int problem_count{0};
    int threadblock_count{0};

    typename EpilogueOutputOp::Params output_op{};

    ElementA ** ptr_A{nullptr};
    ElementB ** ptr_B{nullptr};
    ElementC ** ptr_C{nullptr};
    ElementC ** ptr_D{nullptr};

    typename LayoutA::Stride::LongIndex *lda{nullptr};
    typename LayoutB::Stride::LongIndex *ldb{nullptr};
    typename LayoutC::Stride::LongIndex *ldc{nullptr};
    typename LayoutC::Stride::LongIndex *ldd{nullptr};

    // Only used by device-level operator
    GemmCoord *host_problem_sizes{nullptr};


    //
    // Methods
    //

    /// Default ctor
    Arguments() = default;

    /// Ctor
    CUTLASS_HOST_DEVICE
    Arguments(    
      GemmCoord *problem_sizes,
      int problem_count,
      int threadblock_count,
      typename EpilogueOutputOp::Params output_op,
      ElementA ** ptr_A,
      ElementB ** ptr_B,
      ElementC ** ptr_C,
      ElementC ** ptr_D,
      typename LayoutA::Stride::LongIndex *lda,
      typename LayoutB::Stride::LongIndex *ldb,
      typename LayoutC::Stride::LongIndex *ldc,
      typename LayoutC::Stride::LongIndex *ldd,
      GemmCoord *host_problem_sizes=nullptr
    ): 
      problem_sizes(problem_sizes),
      problem_count(problem_count),
      threadblock_count(threadblock_count),
      output_op(output_op),
      ptr_A(ptr_A),
      ptr_B(ptr_B),
      ptr_C(ptr_C),
      ptr_D(ptr_D),
      lda(lda),
      ldb(ldb),
      ldc(ldc),
      ldd(ldd),
      host_problem_sizes(host_problem_sizes)
    {

    }
  };

  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure
  struct Params {

    typename ProblemVisitor::Params problem_visitor{};
    int threadblock_count{0};

    typename EpilogueOutputOp::Params output_op{};

    ElementA ** ptr_A{nullptr};
    ElementB ** ptr_B{nullptr};
    ElementC ** ptr_C{nullptr};
    ElementC ** ptr_D{nullptr};

    typename LayoutA::Stride::LongIndex *lda{nullptr};
    typename LayoutB::Stride::LongIndex *ldb{nullptr};
    typename LayoutC::Stride::LongIndex *ldc{nullptr};
    typename LayoutC::Stride::LongIndex *ldd{nullptr};

    //
    // Methods
    //

    Params() = default;

    CUTLASS_HOST_DEVICE
    Params(Arguments const &args,
          void *workspace = nullptr,
          int tile_count = 0):
      problem_visitor(args.problem_sizes, args.problem_count, workspace, tile_count),
      threadblock_count(args.threadblock_count),
      output_op(args.output_op),
      ptr_A(args.ptr_A),
      ptr_B(args.ptr_B),
      ptr_C(args.ptr_C),
      ptr_D(args.ptr_D),
      lda(args.lda),
      ldb(args.ldb),
      ldc(args.ldc),
      ldd(args.ldd)
    { 
      printf("[Init Params]: threadblock_count = %d, tile_count = %d, problem_count = %d\n", threadblock_count, tile_count, args.problem_count);
    }

    CUTLASS_HOST_DEVICE
    void update(
      Arguments const &args,
      void *workspace = nullptr,
      int tile_count = 0) {

      problem_visitor = typename ProblemVisitor::Params(args.problem_sizes, args.problem_count,
                                                        workspace, tile_count);
      threadblock_count = args.threadblock_count;
      output_op = args.output_op;
      ptr_A = args.ptr_A;
      ptr_B = args.ptr_B;
      ptr_C = args.ptr_C;
      ptr_D = args.ptr_D;
      lda = args.lda;
      ldb = args.ldb;
      ldc = args.ldc;
      ldd = args.ldd;
    }
  };

  /// Shared memory storage structure
  struct SharedStorage {
    union {
      typename Mma::SharedStorage main_loop;
      typename Epilogue::SharedStorage epilogue;
    } kernel;

    // ProblemVisitor shared storage can't be overlapped with others
    typename ProblemVisitor::SharedStorage problem_visitor;
  };

public:

  //
  // Methods
  //

  CUTLASS_DEVICE
  GemmGrouped() { } 

  /// Determines whether kernel satisfies alignment
  static Status can_implement(cutlass::gemm::GemmCoord const & problem_size) {
    return Status::kSuccess;
  }

  static Status can_implement(Arguments const &args) {
    return Status::kSuccess;
  }

  CUTLASS_DEVICE
  void share_accumulators(
    AccumulatorTile const &accumulator_tile,
    ProblemVisitor const &problem_visitor,
    int block_idx,
    int thread_idx,
    int first_block_idx)
  {
    //AccumulatorTile *accum_tile_workspace = reinterpret_cast<AccumulatorTile *>(params.partials_workspace);
    AccumulatorTile *accum_tile_workspace = reinterpret_cast<AccumulatorTile *>(problem_visitor.partials_ptr);

    int accum_tile_offset = first_block_idx * kThreadCount;

    //if ((block_idx==0||block_idx==1)&&thread_idx==0)
    //  printf("[share_accumulators] Bid: %d, first_block_idx: %d, accum_tile_offset: %d\n", block_idx, first_block_idx, accum_tile_offset);

    if (block_idx == first_block_idx)
    {
      // First peer initializes the workspace partials
      BlockStripedReduceT::store(accum_tile_workspace + accum_tile_offset, accumulator_tile, thread_idx);
    }
    else
    {
      // Subsequent peers atomically accumulate into the workspace partials
      //if (ThreadblockSwizzle::kReductionStrategy == ThreadblockSwizzle::kAtomic)
      //{
      //  // Non-deterministic reduction order: wait for the first peer to have initialized the partials before we add to them
      //  Barrier::wait_lt(params.barrier_workspace, thread_idx, first_block_idx, 1);
      //}
      //else
      //{
      //  // Turnstile reduction order: wait until the previous peer has written
      //  int wait_count = block_idx - first_block_idx;
      //  Barrier::wait_eq(params.barrier_workspace, thread_idx, first_block_idx, wait_count);
      //}

      // TODO which strategy faster???
      //printf("Bid: %d, first_block_idx: %d\n", blockIdx.x, first_block_idx);
      //cudaAssert(false);
      //// atomic
      Barrier::wait_lt(problem_visitor.barrier_ptr, thread_idx, first_block_idx, 1);

      //// otherwise (this is default...)
      // int wait_count = block_idx - first_block_idx;
      // Barrier::wait_eq(problem_visitor.barrier_ptr, thread_idx, first_block_idx, wait_count);

      // Perform reduction in workspace
      BlockStripedReduceT::reduce(accum_tile_workspace + accum_tile_offset, accumulator_tile, thread_idx);
    }

    // Signal our arrival
    // Barrier::arrive_inc(params.barrier_workspace, thread_idx, first_block_idx);
    Barrier::arrive_inc(problem_visitor.barrier_ptr, thread_idx, first_block_idx);
  }

  CUTLASS_DEVICE
  void acquire_accumulators(
    AccumulatorTile &accumulator_tile,
    ProblemVisitor const &problem_visitor,
    int block_idx,
    int thread_idx,
    int first_block_idx)
  {
    //if ((block_idx==0||block_idx==1)&&thread_idx==0)
    //  printf("[acquire_accumulators]: Bid: %d, first_block_idx: %d, num_carry_in: %d\n", block_idx, first_block_idx, block_idx - first_block_idx);

    //AccumulatorTile *accum_tile_workspace = reinterpret_cast<AccumulatorTile *>(params.partials_workspace);
    AccumulatorTile *accum_tile_workspace = reinterpret_cast<AccumulatorTile *>(problem_visitor.partials_ptr);

    // Wait for arrival
    int num_carry_in = block_idx - first_block_idx;
    //Barrier::wait_eq_reset(params.barrier_workspace, thread_idx, first_block_idx, num_carry_in);
    Barrier::wait_eq_reset(problem_visitor.barrier_ptr, thread_idx, first_block_idx, num_carry_in);

    // Load and add peer-partials accumulator tile to local accumulator tile
    int accum_tile_offset = first_block_idx * kThreadCount;
    // BlockStripedReduceT::load_add(accumulator_tile, accum_tile_workspace + accum_tile_offset, thread_idx);
    BlockStripedReduceT::load_add(accumulator_tile, accum_tile_workspace + accum_tile_offset, thread_idx);
  }


  CUTLASS_DEVICE
  void sk_work(ProblemVisitor const &problem_visitor, Params const &params, SharedStorage &shared_storage, int warp_idx, int lane_idx, int thread_idx, int block_idx) {
    //
    // Initialize input iterators
    //
    int const &problem_idx = problem_visitor.sk_tile_work.problem_idx;
    auto const &tile_work = problem_visitor.sk_tile_work;
    using ElementA = typename Mma::IteratorA::Element;
    using LayoutA = typename Mma::IteratorA::Layout;
    using ElementB = typename Mma::IteratorB::Element;
    using LayoutB = typename Mma::IteratorB::Layout;
    using ElementC = typename Epilogue::OutputTileIterator::Element;
    using LayoutC = typename Epilogue::OutputTileIterator::Layout;

    // A
    ElementA *ptr_A = reinterpret_cast<ElementA *>((kTransposed ? params.ptr_B[problem_idx] : params.ptr_A[problem_idx]));
    typename LayoutA::LongIndex ldm_A = (kTransposed ? params.ldb[problem_idx] : params.lda[problem_idx]);

    int m_begin = tile_work.tiled_coord.m() * Mma::Shape::kM;
    //int m_end = params.block_mapping.problem_size.m();
    int m_end = params.problem_visitor.problem_sizes[problem_idx].m();

    typename Mma::IteratorA iterator_A(
      LayoutA(ldm_A),
      ptr_A,
      { m_end, tile_work.k_end },   // extend
      thread_idx,
      { m_begin, tile_work.k_begin }/* offset */);

    // B
    ElementB *ptr_B = reinterpret_cast<ElementB *>((kTransposed ? params.ptr_A[problem_idx] : params.ptr_B[problem_idx]));
    typename LayoutB::LongIndex ldm_B = (kTransposed ? params.lda[problem_idx] : params.ldb[problem_idx]);

    int n_begin = tile_work.tiled_coord.n() * Mma::Shape::kN;
    // int n_end = params.block_mapping.problem_size.n();
    int n_end = params.problem_visitor.problem_sizes[problem_idx].n();
    typename Mma::IteratorB iterator_B(
        LayoutB(ldm_B),
        ptr_B,
        { tile_work.k_end, n_end },
        thread_idx,
        { tile_work.k_begin, n_begin });



    //
    // Matrix multiply phase
    //

    // Initialize accumulators
    typename Mma::FragmentC accumulator_tile;
    accumulator_tile.clear();

    // Initialize MMA abstraction
    Mma mma(
      shared_storage.kernel.main_loop,
      thread_idx,
      warp_idx,
      lane_idx);

    // Perform this tile's range of multiply-accumulate (MAC) iterations
    mma(tile_work.k_iters_remaining, accumulator_tile, iterator_A, iterator_B, accumulator_tile);

    //
    // Cooperative SK peer reduction
    //
    //int first_block_idx = params.block_mapping.get_first_block_idx(tile_work.tile_idx);
    int first_block_idx = problem_visitor.get_first_block_idx(tile_work.tile_idx);
    // if ((blockIdx.x < 5 || blockIdx.x==127) && threadIdx.x == 0) {
    //   printf("[DEBUG-SK] BID: %d, block_iter_begin: %d, block_iter_end: %d, block_iters_remaining: %d, first_block_idx: %d\n", blockIdx.x, problem_visitor.block_iter_begin, problem_visitor.block_iter_end, problem_visitor.block_iters_remaining, first_block_idx);
    // }


    if (!tile_work.tile_finished(problem_visitor.sk_runtime_info.problem_size_k)) {

      // // Non "finishing" SK blocks must share their partial accumulator sums through global scratch workspace
      // share_accumulators(accumulator_tile, block_idx, first_block_idx);
      share_accumulators(accumulator_tile, problem_visitor, block_idx, thread_idx, first_block_idx);
    }
    else
    {
      // DP blocks and "finishing" SK blocks must perform epilogue operations and write the output tile

      if (!tile_work.tile_started())
      {
        // A "finishing" SK block must first aggregate its accumulator partial sums with those shared by peer threadblocks
        // acquire_accumulators(accumulator_tile, block_idx, first_block_idx);
        acquire_accumulators(accumulator_tile, problem_visitor, block_idx, thread_idx, first_block_idx);
      }

      do_epilogue(problem_visitor, 
      params, 
      shared_storage, 
      warp_idx, 
      lane_idx, 
      thread_idx,
      accumulator_tile);
    }
  }

  CUTLASS_DEVICE
  void do_epilogue(
    ProblemVisitor const &problem_visitor,
    Params const &params,
    SharedStorage &shared_storage,
    int warp_idx,
    int lane_idx,
    int thread_idx,
    AccumulatorTile &accumulator_tile)
  {
    using ElementC = typename Epilogue::OutputTileIterator::Element;
    using LayoutC = typename Epilogue::OutputTileIterator::Layout;

    int const &problem_idx = problem_visitor.sk_tile_work.problem_idx;
    auto const &tile_work = problem_visitor.sk_tile_work;

      EpilogueOutputOp output_op(params.output_op);

      ElementC *ptr_C = params.ptr_C[problem_idx];
      ElementC *ptr_D = params.ptr_D[problem_idx];

      LayoutC layout_C(params.ldc[problem_idx]);
      LayoutC layout_D(params.ldd[problem_idx]);

      typename Epilogue::OutputTileIterator::Params params_C(layout_C);
      typename Epilogue::OutputTileIterator::Params params_D(layout_D);

    // Update pointers for batched/array mode(s)
    //if (params.mode == GemmUniversalMode::kBatched) {
    //  ptr_C += tile_work.tiled_coord.k() * params.batch_stride_C;
    //  ptr_D += tile_work.tiled_coord.k() * params.batch_stride_D;
    //}
    //if (params.mode == GemmUniversalMode::kArray) {
    //  ptr_C = static_cast<ElementC * const *>(params.ptr_C)[tile_work.tiled_coord.k()];
    //  ptr_D = static_cast<ElementC * const *>(params.ptr_D)[tile_work.tiled_coord.k()];
    //}

    // Location of this tile in item-coords
    MatrixCoord threadblock_item_begin(
      tile_work.tiled_coord.m() * Mma::Shape::kM,
      tile_work.tiled_coord.n() * Mma::Shape::kN
    );

    // Tile iterator loading from source tensor.
    typename Epilogue::OutputTileIterator iterator_C(
        params_C,
        ptr_C,
        //params.block_mapping.problem_size.mn(),
        params.problem_visitor.problem_sizes[problem_idx].mn(),
        thread_idx,
        threadblock_item_begin);

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D(
        params_D,
        ptr_D,
        //params.block_mapping.problem_size.mn(),
        params.problem_visitor.problem_sizes[problem_idx].mn(),
        thread_idx,
        threadblock_item_begin);

      Epilogue epilogue(
        shared_storage.kernel.epilogue, 
        thread_idx, 
        warp_idx, 
        lane_idx);

      // Execute the epilogue operator to update the destination tensor.
      epilogue(
        output_op, 
        iterator_D, 
        accumulator_tile, 
        iterator_C); 
  }


 
  CUTLASS_DEVICE
  void dp_work(ProblemVisitor const &problem_visitor, Params const &params, SharedStorage &shared_storage, int warp_idx, int lane_idx, int thread_idx, int block_idx) {

    //
    // These types shadow the type-level definitions and support the ability to implement
    // a 'transposed' GEMM that computes the transposed problems.
    //
    using ElementA = typename Mma::IteratorA::Element;
    using LayoutA = typename Mma::IteratorA::Layout;
    using ElementB = typename Mma::IteratorB::Element;
    using LayoutB = typename Mma::IteratorB::Layout;
    using ElementC = typename Epilogue::OutputTileIterator::Element;
    using LayoutC = typename Epilogue::OutputTileIterator::Layout;
    //
    ////
    //

      GemmCoord problem_size  = problem_visitor.problem_size();
      int32_t problem_idx     = problem_visitor.problem_index();
      int32_t threadblock_idx = int32_t(problem_visitor.threadblock_idx());

      GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

      cutlass::gemm::GemmCoord threadblock_offset(
        int(threadblock_idx / grid_shape.n()) * Mma::Shape::kM,
        int(threadblock_idx % grid_shape.n()) * Mma::Shape::kN,
        0);

      // Load element pointers. Exchange pointers and strides if working on the transpose
      ElementA *ptr_A = reinterpret_cast<ElementA *>((kTransposed ? params.ptr_B[problem_idx] : params.ptr_A[problem_idx]));
      typename LayoutA::LongIndex ldm_A = (kTransposed ? params.ldb[problem_idx] : params.lda[problem_idx]);

      ElementB *ptr_B = reinterpret_cast<ElementB *>((kTransposed ? params.ptr_A[problem_idx] : params.ptr_B[problem_idx]));
      typename LayoutB::LongIndex ldm_B = (kTransposed ? params.lda[problem_idx] : params.ldb[problem_idx]);

      // Compute initial location in logical coordinates
      cutlass::MatrixCoord tb_offset_A{
        threadblock_offset.m(),
        0,
      };

      cutlass::MatrixCoord tb_offset_B{
        0,
        threadblock_offset.n()
      };

      // Construct iterators to A and B operands
      typename Mma::IteratorA iterator_A(
        LayoutA(ldm_A),
        ptr_A,
        {problem_size.m(), problem_size.k()},
        thread_idx,
        tb_offset_A);

      typename Mma::IteratorB iterator_B(
        LayoutB(ldm_B),
        ptr_B,
        {problem_size.k(), problem_size.n()},
        thread_idx,
        tb_offset_B);

      typename Mma::FragmentC accumulators;

      accumulators.clear();
      
      // Broadcast the warp_id computed by lane 0 to ensure dependent code
      // is compiled as warp-uniform.
      // int warp_idx = canonical_warp_idx_sync();

      // int lane_idx = threadIdx.x % 32;

      //
      // Matrix multiply phase
      //

      // Construct thread-scoped matrix multiply
      Mma mma(shared_storage.kernel.main_loop, thread_idx, warp_idx, lane_idx);

      // Compute threadblock-scoped matrix multiply-add
      int gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Wait for all threads to finish their epilogue phases from the previous tile.
      __syncthreads();

      // Compute threadblock-scoped matrix multiply-add
      mma(
        gemm_k_iterations, 
        accumulators, 
        iterator_A, 
        iterator_B, 
        accumulators);

      //
      // Epilogue
      //

      EpilogueOutputOp output_op(params.output_op);

      ElementC *ptr_C = params.ptr_C[problem_idx];
      ElementC *ptr_D = params.ptr_D[problem_idx];

      LayoutC layout_C(params.ldc[problem_idx]);
      LayoutC layout_D(params.ldd[problem_idx]);

      typename Epilogue::OutputTileIterator::Params params_C(layout_C);
      typename Epilogue::OutputTileIterator::Params params_D(layout_D);

      // Tile iterator loading from source tensor.
      typename Epilogue::OutputTileIterator iterator_C(
        params_C,
        ptr_C,
        problem_size.mn(),
        thread_idx,
        threadblock_offset.mn()
      );

      // Tile iterator writing to destination tensor.
      typename Epilogue::OutputTileIterator iterator_D(
        params_D,
        ptr_D,
        problem_size.mn(),
        thread_idx,
        threadblock_offset.mn()
      );

      Epilogue epilogue(
        shared_storage.kernel.epilogue, 
        thread_idx, 
        warp_idx, 
        lane_idx);

      // Execute the epilogue operator to update the destination tensor.
      epilogue(
        output_op, 
        iterator_D, 
        accumulators, 
        iterator_C); 
  }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params &params, SharedStorage &shared_storage) {

    //
    // These types shadow the type-level definitions and support the ability to implement
    // a 'transposed' GEMM that computes the transposed problems.
    //
    using ElementA = typename Mma::IteratorA::Element;
    using LayoutA = typename Mma::IteratorA::Layout;
    using ElementB = typename Mma::IteratorB::Element;
    using LayoutB = typename Mma::IteratorB::Layout;
    using ElementC = typename Epilogue::OutputTileIterator::Element;
    using LayoutC = typename Epilogue::OutputTileIterator::Layout;

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;
    int thread_idx = threadIdx.x;
    int block_idx = blockIdx.x;


    //
    // Problem visitor.
    //
    ProblemVisitor problem_visitor(
      params.problem_visitor,
      shared_storage.problem_visitor,
      block_idx);


    while (problem_visitor.next_tile()) {


      if (problem_visitor.is_sk)
        sk_work(problem_visitor, params, shared_storage, warp_idx, lane_idx, thread_idx, block_idx);
      else 
        dp_work(problem_visitor, params, shared_storage, warp_idx, lane_idx, thread_idx, block_idx);

      // dp_work(problem_visitor, params, shared_storage);

      // 
      problem_visitor.advance(gridDim.x);
    }
  }
};

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
