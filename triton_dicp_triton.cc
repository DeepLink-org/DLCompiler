#include "ir.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "dicp/AscendLegalize/Passes.h"
#include "dicp/AutoBlockify/Passes.h"
#include "dicp/Dialect/CommonIR/Passes.h"
#include "dicp/Dialect/TritonDicp/IR/TritonDicpDialect.h"
#include "dicp/DiscreteMaskAccessConversion/Passes.h"
#include "dicp/DynamicCVPipeline/Passes.h"
#include "dicp/TritonAffinityOpt/Passes.h"
#include "dicp/TritonToAnnotation/Passes.h"
#include "dicp/TritonToHFusion/Passes.h"
#include "dicp/TritonToHIVM/Passes.h"
#include "dicp/TritonToLLVM/Passes.h"
#include "dicp/TritonToLinalg/Passes.h"
#include "dicp/TritonToStructured/Passes.h"
#include "dicp/TritonToUnstructure/Passes.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/Instructions.h"

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace mlir;
using namespace mlir::triton::dicp;

// =============================================================================
// DICPNPUIROpBuilder
// =============================================================================

struct DICPNPUIROpBuilder : public TritonOpBuilder {
  std::string target;
  static constexpr char kTarget910_95[] = "Ascend910_95";
  static constexpr char kTarget950[] = "Ascend950";

  explicit DICPNPUIROpBuilder(MLIRContext *context, std::string target = "")
      : TritonOpBuilder(context), target(target) {}

  bool is_910_95() const {
    constexpr size_t kLen910 = sizeof(kTarget910_95) - 1;
    bool match_910 = target.size() >= kLen910 &&
                     target.compare(0, kLen910, kTarget910_95) == 0;

    constexpr size_t kLen950 = sizeof(kTarget950) - 1;
    bool match_950 =
        target.size() >= kLen950 && target.compare(0, kLen950, kTarget950) == 0;

    return match_910 || match_950;
  }
};

namespace {

MLIRContext *gDefaultDICPContext = nullptr;

MLIRContext *resolveContext(const py::object &contextObj) {
  if (!contextObj.is_none()) {
    return &py::cast<MLIRContext &>(contextObj);
  }
  if (gDefaultDICPContext) {
    return gDefaultDICPContext;
  }
  throw std::invalid_argument(
      "No default MLIR context. Pass context explicitly or call "
      "dicp_ir.load_dialects(context) first.");
}

struct ModeAndPipes {
  hivm::SyncBlockModeAttr modeAttr = {};
  hivm::PipeAttr cubePipe = {};
  hivm::PipeAttr vectorPipe = {};
};

hivm::TCoreTypeAttr GetCore(MLIRContext *ctx, llvm::StringRef opName,
                            llvm::StringRef sender) {
  hivm::TCoreTypeAttr core;
  if (sender == "cube") {
    if (opName == "sync_block_set")
      core = hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::CUBE);
    else
      core = hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::VECTOR);
  } else {
    if (sender != "vector") {
      throw std::runtime_error(
          "sync_block_set/wait only supports 'cube' or 'vector' as sender");
    }
    if (opName == "sync_block_set")
      core = hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::VECTOR);
    else
      core = hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::CUBE);
  }
  return core;
}

void buildSyncBlockOp(DICPNPUIROpBuilder &self, const std::string &opNameSnake,
                      std::string &sender, std::string &receiver, Value id,
                      hivm::PIPE senderPipe, hivm::PIPE receiverPipe) {
  auto *ctx = self.getBuilder().getContext();
  hivm::TCoreTypeAttr coreAttr = GetCore(ctx, opNameSnake, sender);
  hivm::PipeAttr prodPipe = hivm::PipeAttr::get(ctx, senderPipe);
  hivm::PipeAttr consPipe = hivm::PipeAttr::get(ctx, receiverPipe);
  const size_t I64 = 64;
  auto i64Ty = IntegerType::get(ctx, I64);
  Value idI64 = id;
  if (!id.getType().isInteger(I64)) {
    idI64 = mlir::convertScalarToDtype(self.getBuilder(), id.getLoc(), id,
                                       i64Ty, true);
  }
  if (opNameSnake == "sync_block_set") {
    self.create<hivm::SyncBlockSetOp>(coreAttr, prodPipe, consPipe, idI64);
  } else if (opNameSnake == "sync_block_wait") {
    self.create<hivm::SyncBlockWaitOp>(coreAttr, prodPipe, consPipe, idI64);
  } else {
    throw std::runtime_error("Unsupported operation name for SyncBlockOp");
  }
}

ModeAndPipes GetSyncBlockModeAndPipes(MLIRContext *ctx,
                                      const std::string &mode) {
  hivm::SyncBlockModeAttr modeAttr = {};
  hivm::PipeAttr cubePipe = {};
  hivm::PipeAttr vectorPipe = {};

  if (mode == "all_cube") {
    modeAttr = hivm::SyncBlockModeAttr::get(ctx, hivm::SyncBlockMode::ALL_CUBE);
    cubePipe = hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_ALL);
    vectorPipe = hivm::PipeAttr{};
  } else if (mode == "all_vector") {
    modeAttr =
        hivm::SyncBlockModeAttr::get(ctx, hivm::SyncBlockMode::ALL_VECTOR);
    cubePipe = hivm::PipeAttr{};
    vectorPipe = hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_ALL);
  } else if (mode == "all") {
    modeAttr = hivm::SyncBlockModeAttr::get(ctx, hivm::SyncBlockMode::ALL);
    cubePipe = hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_ALL);
    vectorPipe = hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_ALL);
  } else if (mode == "all_sub_vector") {
    modeAttr =
        hivm::SyncBlockModeAttr::get(ctx, hivm::SyncBlockMode::ALL_SUB_VECTOR);
    cubePipe = hivm::PipeAttr{};
    vectorPipe = hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_ALL);
  } else {
    llvm::report_fatal_error(
        llvm::StringRef("Invalid sync-block mode: " + mode));
  }
  return {modeAttr, cubePipe, vectorPipe};
}

} // namespace

// =============================================================================
// init_dicp_ir: IR builder bindings (merged)
// =============================================================================

void init_dicp_ir(py::module &&m) {
  // --- AffineExpr bindings ---
  auto affineExprClass =
      py::class_<AffineExpr>(m, "affine_expr", py::module_local());
  affineExprClass
      .def("__str__",
           [](AffineExpr self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return os.str();
           })
      .def("__repr__",
           [](AffineExpr self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return "<affine_expr " + os.str() + ">";
           })
      .def("is_symbolic_or_constant", &AffineExpr::isSymbolicOrConstant)
      .def("is_pure_affine", &AffineExpr::isPureAffine)
      .def("is_function_of_dim", &AffineExpr::isFunctionOfDim)
      .def("compose",
           [](AffineExpr self, AffineMap map) { return self.compose(map); })
      .def("get_largest_known_divisor", &AffineExpr::getLargestKnownDivisor)
      .def("floordiv", [](AffineExpr self,
                          AffineExpr other) { return self.floorDiv(other); })
      .def("ceildiv", [](AffineExpr self,
                         AffineExpr other) { return self.ceilDiv(other); })
      .def("mod",
           [](AffineExpr self, AffineExpr other) { return self % other; })
      .def("__hash__",
           [](AffineExpr self) {
             return py::int_(static_cast<uint64_t>(mlir::hash_value(self)));
           })
      .def("__eq__", [](AffineExpr lhs, AffineExpr rhs) { return lhs == rhs; })
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * py::self)
      .def(py::self % py::self);
  affineExprClass
      .def_static(
          "get_constant",
          [](int64_t val, py::object contextObj) {
            auto *context = resolveContext(contextObj);
            return getAffineConstantExpr(val, context);
          },
          py::arg("value"), py::arg("context") = py::none())
      .def_static(
          "get_dim",
          [](uint32_t pos, py::object contextObj) {
            auto *context = resolveContext(contextObj);
            return getAffineDimExpr(pos, context);
          },
          py::arg("pos"), py::arg("context") = py::none())
      .def_static(
          "get_symbol",
          [](uint32_t pos, py::object contextObj) {
            auto *context = resolveContext(contextObj);
            return getAffineSymbolExpr(pos, context);
          },
          py::arg("pos"), py::arg("context") = py::none());

  py::class_<AffineConstantExpr, AffineExpr>(m, "affine_constant_expr",
                                             py::module_local())
      .def("get_value", &AffineConstantExpr::getValue);
  py::class_<AffineDimExpr, AffineExpr>(m, "affine_dim_expr",
                                        py::module_local())
      .def("get_position", &AffineDimExpr::getPosition);
  py::class_<AffineSymbolExpr, AffineExpr>(m, "affine_symbol_expr",
                                           py::module_local())
      .def("get_position", &AffineSymbolExpr::getPosition);
  py::class_<AffineBinaryOpExpr, AffineExpr>(m, "affine_binary_op_expr",
                                             py::module_local())
      .def("get_lhs", &AffineBinaryOpExpr::getLHS)
      .def("get_rhs", &AffineBinaryOpExpr::getRHS);

  // --- AffineMap bindings ---
  auto affineMapClass =
      py::class_<AffineMap>(m, "affine_map", py::module_local());
  affineMapClass
      .def("__str__",
           [](AffineMap &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return os.str();
           })
      .def("__repr__",
           [](AffineMap &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return "<affine_map " + os.str() + ">";
           })
      .def("is_identity", &AffineMap::isIdentity)
      .def("is_permutation", &AffineMap::isPermutation)
      .def("get_num_dims", &AffineMap::getNumDims)
      .def("get_num_symbols", &AffineMap::getNumSymbols)
      .def("get_num_results", &AffineMap::getNumResults)
      .def("is_empty", &AffineMap::isEmpty)
      .def("is_single_constant", &AffineMap::isSingleConstant)
      .def("is_constant", &AffineMap::isConstant)
      .def("get_constant_result",
           [](AffineMap &self) -> int64_t {
             if (!self.isSingleConstant()) {
               throw std::runtime_error(
                   "affine map is not a single constant map");
             }
             return self.getSingleConstantResult();
           })
      .def("get_result",
           [](AffineMap &self, uint32_t pos) {
             if (pos >= self.getNumResults()) {
               throw py::index_error("result index out of range");
             }
             return self.getResult(pos);
           })
      .def("get_sub_map",
           [](AffineMap &self, const std::vector<unsigned> &resultPos) {
             return self.getSubMap(resultPos);
           })
      .def("replace",
           [](AffineMap &self, AffineExpr expr, AffineExpr replacement,
              uint32_t numResultDims, uint32_t numResultSymbols) {
             return self.replace(expr, replacement, numResultDims,
                                 numResultSymbols);
           })
      .def("compose",
           [](AffineMap &self, AffineMap map) { return self.compose(map); })
      .def("get_results",
           [](AffineMap &self) -> std::vector<AffineExpr> {
             auto results = self.getResults();
             return std::vector<AffineExpr>(results.begin(), results.end());
           })
      .def("__hash__",
           [](AffineMap &self) {
             return py::int_(static_cast<uint64_t>(mlir::hash_value(self)));
           })
      .def("__eq__", [](AffineMap &lhs, AffineMap &rhs) { return lhs == rhs; })
      .def("inverse_permutation",
           [](AffineMap &self) -> py::object {
             if (!self.isPermutation()) {
               throw py::value_error(
                   "AffineMap must be a valid permutation to compute inverse");
             }
             AffineMap inverse = mlir::inversePermutation(self);
             if (!inverse) {
               throw py::value_error("Failed to compute inverse permutation");
             }
             return py::cast(inverse);
           })
      .def("to_dict", [](AffineMap &self) -> py::dict {
        py::list results;
        for (AffineExpr result : self.getResults()) {
          if (auto dimExpr = dyn_cast<AffineDimExpr>(result)) {
            results.append(dimExpr.getPosition());
          } else {
            std::string exprStr;
            llvm::raw_string_ostream os(exprStr);
            result.print(os);
            results.append(py::str(exprStr));
          }
        }
        py::dict ret;
        ret["num_dims"] = self.getNumDims();
        ret["num_symbols"] = self.getNumSymbols();
        ret["results"] = std::move(results);
        return ret;
      });
  affineMapClass
      .def_static(
          "get",
          [](int64_t numDims, int64_t numSymbols, const py::iterable &resultsIn,
             py::object contextObj) -> AffineMap {
            MLIRContext *context = nullptr;
            if (numDims < 0 || numSymbols < 0) {
              throw std::invalid_argument(
                  "num_dims and num_symbols must be non-negative");
            }
            llvm::SmallVector<AffineExpr> results;
            for (const auto &item : resultsIn) {
              if (py::isinstance<AffineExpr>(item)) {
                auto expr = py::cast<AffineExpr>(item);
                if (!context) {
                  context = expr.getContext();
                }
                results.push_back(expr);
                continue;
              }
              if (py::isinstance<py::int_>(item)) {
                if (!context) {
                  context = resolveContext(contextObj);
                }
                int64_t pos = py::cast<int64_t>(item);
                if (pos < 0 || pos >= numDims) {
                  throw std::invalid_argument(
                      "result dim index is out of range for num_dims");
                }
                results.push_back(getAffineDimExpr(pos, context));
                continue;
              }
              throw std::invalid_argument(
                  "results must contain affine_expr or int dim indices");
            }
            if (!context) {
              context = resolveContext(contextObj);
            }
            return AffineMap::get(numDims, numSymbols, results, context);
          },
          py::arg("num_dims"), py::arg("num_symbols"), py::arg("result_dims"),
          py::arg("context") = py::none())
      .def_static(
          "get_identity",
          [](int64_t numDims, py::object contextObj) -> AffineMap {
            auto *context = resolveContext(contextObj);
            if (numDims < 0) {
              throw std::invalid_argument("num_dims must be non-negative");
            }
            return AffineMap::getMultiDimIdentityMap(numDims, context);
          },
          py::arg("num_dims"), py::arg("context") = py::none())
      .def_static(
          "get_minor_identity",
          [](int64_t dims, int64_t results, py::object contextObj) {
            auto *context = resolveContext(contextObj);
            if (dims < 0 || results < 0) {
              throw std::invalid_argument("dims/results must be non-negative");
            }
            return AffineMap::getMinorIdentityMap(dims, results, context);
          },
          py::arg("dims"), py::arg("results"), py::arg("context") = py::none())
      .def_static(
          "get_empty",
          [](py::object contextObj) {
            auto *context = resolveContext(contextObj);
            return AffineMap::get(0, 0, {}, context);
          },
          py::arg("context") = py::none())
      .def_static(
          "get_permutation",
          [](const std::vector<unsigned> &permutation, py::object contextObj) {
            auto *context = resolveContext(contextObj);
            return AffineMap::getPermutationMap(permutation, context);
          },
          py::arg("permutation"), py::arg("context") = py::none())
      .def_static(
          "get_constant",
          [](int64_t value, py::object contextObj) {
            auto *context = resolveContext(contextObj);
            return AffineMap::getConstantMap(value, context);
          },
          py::arg("value"), py::arg("context") = py::none());

  // --- hivm enums ---
  py::enum_<hivm::AddressSpace>(m, "AddressSpace", py::module_local())
      .value("L1", hivm::AddressSpace::L1)
      .value("UB", hivm::AddressSpace::UB)
      .value("L0A", hivm::AddressSpace::L0A)
      .value("L0B", hivm::AddressSpace::L0B)
      .value("L0C", hivm::AddressSpace::L0C)
      .export_values();

  py::enum_<hivm::TCoreType>(m, "CoreType", py::module_local())
      .value("CUBE", hivm::TCoreType::CUBE)
      .value("VECTOR", hivm::TCoreType::VECTOR)
      .value("CUBE_OR_VECTOR", hivm::TCoreType::CUBE_OR_VECTOR)
      .value("CUBE_AND_VECTOR", hivm::TCoreType::CUBE_AND_VECTOR)
      .export_values();

  py::enum_<hivm::PIPE>(m, "PIPE", py::module_local())
      .value("PIPE_S", hivm::PIPE::PIPE_S)
      .value("PIPE_V", hivm::PIPE::PIPE_V)
      .value("PIPE_M", hivm::PIPE::PIPE_M)
      .value("PIPE_MTE1", hivm::PIPE::PIPE_MTE1)
      .value("PIPE_MTE2", hivm::PIPE::PIPE_MTE2)
      .value("PIPE_MTE3", hivm::PIPE::PIPE_MTE3)
      .value("PIPE_ALL", hivm::PIPE::PIPE_ALL)
      .value("PIPE_FIX", hivm::PIPE::PIPE_FIX)
      .export_values();

  py::enum_<hivm::VFMode>(m, "MODE", py::module_local())
      .value("SIMD", hivm::VFMode::SIMD)
      .value("SIMT", hivm::VFMode::SIMT)
      .value("MIX", hivm::VFMode::MIX)
      .export_values();

  py::enum_<hivm::IteratorType>(m, "IteratorType", py::module_local())
      .value("Parallel", hivm::IteratorType::kParallel)
      .value("Broadcast", hivm::IteratorType::kBroadcast)
      .value("Transpose", hivm::IteratorType::kTranspose)
      .value("Reduction", hivm::IteratorType::kReduction)
      .value("Interleave", hivm::IteratorType::kInterleave)
      .value("Deinterleave", hivm::IteratorType::kDeinterleave)
      .value("Inverse", hivm::IteratorType::kInverse)
      .value("Pad", hivm::IteratorType::kPad)
      .value("Concat", hivm::IteratorType::kConcat)
      .value("Gather", hivm::IteratorType::kGather)
      .value("Cumulative", hivm::IteratorType::kCumulative)
      .value("Opaque", hivm::IteratorType::kOpaque)
      .export_values();

  py::enum_<hivm::FixpipeDMAMode>(m, "FixpipeDMAMode", py::module_local())
      .value("NZ2DN", hivm::FixpipeDMAMode::NZ2DN)
      .value("NZ2ND", hivm::FixpipeDMAMode::NZ2ND)
      .value("NZ2NZ", hivm::FixpipeDMAMode::NZ2NZ)
      .export_values();

  py::enum_<hivm::FixpipeDualDstMode>(m, "FixpipeDualDstMode",
                                      py::module_local())
      .value("NO_DUAL", hivm::FixpipeDualDstMode::NO_DUAL)
      .value("COLUMN_SPLIT", hivm::FixpipeDualDstMode::COLUMN_SPLIT)
      .value("ROW_SPLIT", hivm::FixpipeDualDstMode::ROW_SPLIT)
      .export_values();

  py::enum_<hivm::FixpipePreQuantMode>(m, "FixpipePreQuantMode",
                                       py::module_local())
      .value("NO_QUANT", hivm::FixpipePreQuantMode::NO_QUANT)
      .value("F322BF16", hivm::FixpipePreQuantMode::F322BF16)
      .value("F322F16", hivm::FixpipePreQuantMode::F322F16)
      .value("S322I8", hivm::FixpipePreQuantMode::S322I8)
      .export_values();

  py::enum_<hivm::FixpipePreReluMode>(m, "FixpipePreReluMode",
                                      py::module_local())
      .value("LEAKY_RELU", hivm::FixpipePreReluMode::LEAKY_RELU)
      .value("NO_RELU", hivm::FixpipePreReluMode::NO_RELU)
      .value("NORMAL_RELU", hivm::FixpipePreReluMode::NORMAL_RELU)
      .value("P_RELU", hivm::FixpipePreReluMode::P_RELU)
      .export_values();

  py::enum_<hivm::DataLayout>(m, "DataLayout", py::module_local())
      .value("nZ", hivm::DataLayout::nZ)
      .value("zN", hivm::DataLayout::zN)
      .export_values();

  m.def("load_dialects", [](MLIRContext &context) {
    gDefaultDICPContext = &context;
    DialectRegistry registry;
    registry.insert<annotation::AnnotationDialect, mlir::hivm::HIVMDialect,
                    hacc::HACCDialect, triton::dicp::TritonDicpDialect,
                    scope::ScopeDialect>();
    mlir::func::registerInlinerExtension(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  // --- dicp_npu_ir_builder class ---
  py::class_<DICPNPUIROpBuilder, TritonOpBuilder>(
      m, "dicp_npu_ir_builder", py::module_local(), py::dynamic_attr())
      .def(py::init<MLIRContext *, std::string>(), py::arg("context"),
           py::arg("target") = "")
      .def("get_int_attr",
           [](DICPNPUIROpBuilder &self, int64_t value) -> Attribute {
             return IntegerAttr::get(self.getBuilder().getI64Type(), value);
           })
      .def("get_str_array_attr",
           [](DICPNPUIROpBuilder &self,
              const std::vector<std::string> &values) -> Attribute {
             auto *ctx = self.getBuilder().getContext();
             llvm::SmallVector<Attribute> attrs;
             attrs.reserve(values.size());
             for (const auto &v : values)
               attrs.push_back(self.getBuilder().getStringAttr(v));
             return ArrayAttr::get(ctx, attrs);
           })
      .def("get_i64_array_attr",
           [](DICPNPUIROpBuilder &self,
              const std::vector<int64_t> &values) -> Attribute {
             return self.getBuilder().getI64ArrayAttr(values);
           })
      .def(
          "get_core_type_attr",
          [](DICPNPUIROpBuilder &self, hivm::TCoreType core_type) -> Attribute {
            return self.getBuilder().getAttr<hivm::TCoreTypeAttr>(core_type);
          })
      .def("get_pipe_attr",
           [](DICPNPUIROpBuilder &self, hivm::PIPE pipe) -> Attribute {
             return self.getBuilder().getAttr<hivm::PipeAttr>(pipe);
           })
      .def("get_vf_mode_attr",
           [](DICPNPUIROpBuilder &self, hivm::VFMode mode) -> Attribute {
             return self.getBuilder().getAttr<hivm::VFModeAttr>(mode);
           })
      .def("get_iterator_types_attr",
           [](DICPNPUIROpBuilder &self,
              const std::vector<hivm::IteratorType> &array) {
             auto attrs = llvm::to_vector(
                 llvm::map_range(array, [&self](hivm::IteratorType type) {
                   return cast<Attribute>(
                       self.getBuilder().getAttr<hivm::IteratorTypeAttr>(type));
                 }));
             return self.getBuilder().getArrayAttr(attrs);
           })
      .def("get_t_core_type_attr_name",
           [](DICPNPUIROpBuilder &self) -> std::string {
             return hivm::TCoreTypeAttr::name.str();
           })
      .def("get_t_core_type_cube_attr",
           [](DICPNPUIROpBuilder &self) -> Attribute {
             return hivm::TCoreTypeAttr::get(self.getBuilder().getContext(),
                                             hivm::TCoreType::CUBE);
           })
      .def("get_t_core_type_vector_attr",
           [](DICPNPUIROpBuilder &self) -> Attribute {
             return hivm::TCoreTypeAttr::get(self.getBuilder().getContext(),
                                             hivm::TCoreType::VECTOR);
           })
      .def("parse_attr",
           [](TritonOpBuilder &self, std::string value) -> Attribute {
             auto *ctx = self.getBuilder().getContext();
             ctx->allowUnregisteredDialects();
             return mlir::parseAttribute(value, ctx);
           })
      .def("get_affine_map_attr",
           [](DICPNPUIROpBuilder &self, AffineMap affineMap) -> Attribute {
             return AffineMapAttr::get(affineMap);
           })
      .def("get_affine_map_array_attr",
           [](DICPNPUIROpBuilder &self,
              const std::vector<AffineMap> &affineMaps) -> Attribute {
             auto *ctx = self.getBuilder().getContext();
             llvm::SmallVector<Attribute> attrs;
             attrs.reserve(affineMaps.size());
             for (const auto &map : affineMaps) {
               attrs.push_back(AffineMapAttr::get(map));
             }
             return ArrayAttr::get(ctx, attrs);
           })
      .def("get_buffer_ty_with_affine_map",
           [](DICPNPUIROpBuilder &self, std::vector<int64_t> &shape,
              Type &elementType, AffineMap affineMap,
              const Attribute &memorySpace) -> Type {
             auto layout = AffineMapAttr::get(affineMap);
             return MemRefType::get(shape, elementType, layout, memorySpace);
           })
      .def("create_fixpipe",
           [](DICPNPUIROpBuilder &self, Value src, py::object dst_obj,
              hivm::FixpipeDMAMode dma_mode,
              hivm::FixpipeDualDstMode dual_dst_mode,
              hivm::FixpipePreQuantMode pre_quant_mode,
              hivm::FixpipePreReluMode pre_relu_mode) -> py::object {
             if (!dyn_cast<RankedTensorType>(src.getType())) {
               llvm_unreachable("src is not of RankedTensorType");
             }
             auto *ctx = self.getBuilder().getContext();
             auto loc = self.getLastLoc();
             Value dstValue;
             bool needCreateDst = dst_obj.is_none();
             if (needCreateDst) {
               auto srcType = dyn_cast<RankedTensorType>(src.getType());
               auto srcShape = srcType.getShape();
               llvm::SmallVector<int64_t> dstShape(srcShape.begin(),
                                                   srcShape.end());
               if (dual_dst_mode == hivm::FixpipeDualDstMode::ROW_SPLIT) {
                 if (dstShape.size() >= 1 && dstShape[0] > 0) {
                   dstShape[0] = dstShape[0] / 2;
                 }
               } else if (dual_dst_mode ==
                          hivm::FixpipeDualDstMode::COLUMN_SPLIT) {
                 if (dstShape.size() >= 2 && dstShape[1] > 0) {
                   dstShape[1] = dstShape[1] / 2;
                 }
               }
               auto dstType =
                   RankedTensorType::get(dstShape, srcType.getElementType());
               auto emptyTensor = self.create<tensor::EmptyOp>(
                   dstType.getShape(), dstType.getElementType());
               dstValue = emptyTensor.getResult();
             } else {
               dstValue = py::cast<Value>(dst_obj);
               if (!dyn_cast<ShapedType>(dstValue.getType())) {
                 llvm_unreachable("dst is not of ShapedType");
               }
             }
             auto dma_mode_attr =
                 mlir::hivm::FixpipeDMAModeAttr::get(ctx, dma_mode);
             auto dual_dst_mode_attr =
                 mlir::hivm::FixpipeDualDstModeAttr::get(ctx, dual_dst_mode);
             auto pre_quant_mode_attr =
                 mlir::hivm::FixpipePreQuantModeAttr::get(ctx, pre_quant_mode);
             auto pre_relu_mode_attr =
                 mlir::hivm::FixpipePreReluModeAttr::get(ctx, pre_relu_mode);
             auto channel_split = BoolAttr::get(ctx, false);
             if (needCreateDst) {
               return py::cast<Value>(
                   self.create<hivm::FixpipeOp>(
                           mlir::TypeRange{dstValue.getType()}, src, dstValue,
                           dma_mode_attr, dual_dst_mode_attr,
                           pre_quant_mode_attr, pre_relu_mode_attr,
                           channel_split)
                       .getResult(0));
             } else {
               self.create<hivm::FixpipeOp>(mlir::TypeRange{}, src, dstValue,
                                            dma_mode_attr, dual_dst_mode_attr,
                                            pre_quant_mode_attr,
                                            pre_relu_mode_attr, channel_split);
               return py::none();
             }
           })
      .def("create_bind_buffer",
           [](TritonOpBuilder &self, Value &src, Value &alloc) -> void {
             auto ctx = self.getBuilder().getContext();
             auto bind = StringAttr::get(ctx, "bind_buffer");
             self.create<annotation::MarkOp>(src, ValueRange{alloc},
                                             ArrayAttr::get(ctx, bind));
           })
      .def("create_debug_barrier",
           [](TritonOpBuilder &self, Value &ptr, const std::string &attrKey,
              Attribute &attrVal) {
             auto annotationOp = self.create<annotation::MarkOp>(ptr);
             annotationOp->setAttr(self.getBuilder().getStringAttr(attrKey),
                                   attrVal);
           })
      .def("create_custom_op",
           [](DICPNPUIROpBuilder &self, const std::string &name,
              const py::dict &attrs, const std::vector<Value> &ins,
              const std::vector<Value> &outs,
              const std::vector<py::dict> &arg_attrs) -> std::vector<Value> {
             ValueRange inputs{ins};
             ValueRange outputs{outs};
             ValueRange temp_buffers{};
             TypeRange res_types{outputs};
             auto op = self.create<hivm::CustomOp>(res_types, name, inputs,
                                                   outputs, temp_buffers);
             for (auto &attr : attrs) {
               std::string attr_name = py::cast<std::string>(attr.first);
               Attribute attr_value = py::cast<Attribute>(attr.second);
               op->setAttr(attr_name, attr_value);
             }
             SmallVector<Attribute> dictAttrs(arg_attrs.size());
             Attribute emptyDict = self.getBuilder().getDictionaryAttr({});
             for (const auto &[idx, attrs] : llvm::enumerate(arg_attrs)) {
               if (idx >= op.getNumOperands())
                 continue;
               if (attrs.is_none()) {
                 dictAttrs[idx] = emptyDict;
                 continue;
               }
               llvm::SmallVector<NamedAttribute> namedAttrs;
               for (const auto &attr : attrs) {
                 std::string attr_name = py::cast<std::string>(attr.first);
                 Attribute attr_value = py::cast<Attribute>(attr.second);
                 namedAttrs.push_back(NamedAttribute(
                     self.getBuilder().getStringAttr(attr_name), attr_value));
               }
               dictAttrs[idx] = self.getBuilder().getDictionaryAttr(namedAttrs);
             }
             ArrayAttr arg_attrs_array =
                 self.getBuilder().getArrayAttr(dictAttrs);
             op->setAttr("arg_attrs", arg_attrs_array);
             auto results = op->getResults();
             return std::vector<Value>(results.begin(), results.end());
           })
      .def("create_scope_op",
           [](DICPNPUIROpBuilder &self, py::dict &scopeAttrs,
              std::vector<Type> resultTypes) -> OpState {
             llvm::SmallVector<NamedAttribute> attrs;
             for (auto item : scopeAttrs) {
               std::string key = py::cast<std::string>(item.first);
               Attribute value = py::cast<Attribute>(item.second);
               attrs.push_back(
                   NamedAttribute(self.getBuilder().getStringAttr(key), value));
             }
             auto scopeOp = self.create<scope::ScopeOp>(TypeRange(resultTypes));
             scopeOp->setAttrs(attrs);
             return OpState(scopeOp);
           })
      .def(
          "scope_return",
          [](DICPNPUIROpBuilder &self, std::vector<Value> operands) -> OpState {
            return self.create<scope::ReturnOp>(ValueRange(operands));
          })
      .def("sync_block_set",
           [](DICPNPUIROpBuilder &self, std::string &sender,
              std::string &receiver, Value id, hivm::PIPE senderPipe,
              hivm::PIPE receiverPipe) -> void {
             buildSyncBlockOp(self, "sync_block_set", sender, receiver, id,
                              senderPipe, receiverPipe);
           })
      .def("sync_block_wait",
           [](DICPNPUIROpBuilder &self, std::string &sender,
              std::string &receiver, Value id, hivm::PIPE senderPipe,
              hivm::PIPE receiverPipe) -> void {
             buildSyncBlockOp(self, "sync_block_wait", sender, receiver, id,
                              senderPipe, receiverPipe);
           })
      .def("get_target_attribute",
           [](DICPNPUIROpBuilder &self,
              hivm::AddressSpace &addressSpace) -> Attribute {
             return hivm::AddressSpaceAttr::get(self.getBuilder().getContext(),
                                                addressSpace);
           })
      .def("create_get_sub_vec_id",
           [](DICPNPUIROpBuilder &self) -> Value {
             auto subBlockIdxOp = self.create<hivm::GetSubBlockIdxOp>();
             auto moduleOp = subBlockIdxOp->getParentOfType<ModuleOp>();
             auto *ctx = self.getBuilder().getContext();
             moduleOp->setAttr("hivm.disable_auto_tile_and_bind_subblock",
                               mlir::UnitAttr::get(ctx));
             return subBlockIdxOp;
           })
      .def("sync_block_all",
           [](DICPNPUIROpBuilder &self, std::string &mode, int id) -> void {
             auto *ctx = self.getBuilder().getContext();
             auto [modeAttr, cubePipe, vectorPipe] =
                 GetSyncBlockModeAndPipes(ctx, mode);
             mlir::IndexType indexType = mlir::IndexType::get(ctx);
             mlir::IntegerAttr indexAttribute =
                 mlir::IntegerAttr::get(indexType, static_cast<int64_t>(id));
             self.create<hivm::SyncBlockOp>(
                 modeAttr, indexAttribute, mlir::Value{}, cubePipe, vectorPipe);
           })
      .def("is_910_95",
           [](DICPNPUIROpBuilder &self) -> bool { return self.is_910_95(); })
      .def("create_copy_buffer",
           [](DICPNPUIROpBuilder &self, Value src, Value dst) {
             self.create<hivm::CopyOp>(mlir::TypeRange{}, src, dst);
           })
      .def("create_copy_tensor",
           [](DICPNPUIROpBuilder &self, Value src, Value dst) {
             return self
                 .create<hivm::CopyOp>(mlir::TypeRange{dst.getType()}, src, dst)
                 .getResult(0);
           })
      .def("create_convert_layout",
           [](DICPNPUIROpBuilder &self, Value src, Type memrefType) -> Value {
             auto *ctx = self.getBuilder().getContext();
             return self
                 .create<hivm::ConvertLayoutOp>(
                     memrefType, src,
                     hivm::DataLayoutAttr::get(ctx, hivm::DataLayout::ND),
                     hivm::DataLayoutAttr::get(ctx, hivm::DataLayout::ND))
                 .getResult();
           });

  // --- DICP extension methods on TritonOpBuilder (via getBuilderClass) ---
  auto *builder_cls = ir::getBuilderClass();
  if (builder_cls) {
    builder_cls
        ->def("create_extract_slice",
              [](TritonOpBuilder &self, Value &ful,
                 std::vector<Value> &offs_vec, std::vector<int> &sizs_vec,
                 std::vector<int> &strd_vec) -> Value {
                self.getContext()
                    ->getOrLoadDialect<mlir::tensor::TensorDialect>();
                llvm::SmallVector<Value> offsets;
                llvm::SmallVector<int64_t> staticOffsets;
                for (const auto &o : offs_vec) {
                  auto oTy = o.getType();
                  if (!oTy.isIndex()) {
                    auto v = self.create<arith::IndexCastOp>(
                        self.getBuilder().getIndexType(), o);
                    offsets.push_back(v);
                  } else {
                    offsets.push_back(o);
                  }
                  staticOffsets.push_back(ShapedType::kDynamic);
                }
                llvm::SmallVector<Value> sizes;
                llvm::SmallVector<int64_t> staticSizes;
                llvm::SmallVector<int64_t> retSizes;
                for (const auto &s : sizs_vec) {
                  staticSizes.push_back(s);
                  retSizes.push_back(s);
                }
                llvm::SmallVector<Value> strides;
                llvm::SmallVector<int64_t> staticStrides;
                for (const auto &s : strd_vec) {
                  auto v = self.create<arith::ConstantIndexOp>(s);
                  strides.push_back(v);
                  staticStrides.push_back(ShapedType::kDynamic);
                }
                auto retTy = RankedTensorType::get(
                    retSizes,
                    cast<RankedTensorType>(ful.getType()).getElementType());
                return self.create<tensor::ExtractSliceOp>(
                    retTy, ful, offsets, sizes, strides, staticOffsets,
                    staticSizes, staticStrides);
              })
        .def("create_insert_slice",
             [](TritonOpBuilder &self, Value &ful, Value &sub,
                std::vector<Value> &offs_vec, std::vector<int> &sizs_vec,
                std::vector<int> &strd_vec) -> Value {
               self.getContext()
                   ->getOrLoadDialect<mlir::tensor::TensorDialect>();
               llvm::SmallVector<Value> offsets;
               llvm::SmallVector<int64_t> staticOffsets;
               for (const auto &o : offs_vec) {
                 auto oTy = o.getType();
                 if (!oTy.isIndex()) {
                   auto v = self.create<arith::IndexCastOp>(
                       self.getBuilder().getIndexType(), o);
                   offsets.push_back(v);
                 } else {
                   offsets.push_back(o);
                 }
                 staticOffsets.push_back(ShapedType::kDynamic);
               }
               llvm::SmallVector<Value> sizes;
               llvm::SmallVector<int64_t> staticSizes;
               llvm::SmallVector<int64_t> retSizes;
               for (const auto &s : sizs_vec) {
                 staticSizes.push_back(s);
                 retSizes.push_back(s);
               }
               llvm::SmallVector<Value> strides;
               llvm::SmallVector<int64_t> staticStrides;
               for (const auto &s : strd_vec) {
                 auto v = self.create<arith::ConstantIndexOp>(s);
                 strides.push_back(v);
                 staticStrides.push_back(ShapedType::kDynamic);
               }
               auto retTy = RankedTensorType::get(
                   retSizes,
                   cast<RankedTensorType>(ful.getType()).getElementType());
               auto ret = self.create<tensor::InsertSliceOp>(
                   sub, ful, offsets, sizes, strides, staticOffsets,
                   staticSizes, staticStrides);
               return ret;
             })
        .def("create_annotation_mark",
             [](TritonOpBuilder &self, Value &ptr, const std::string &attrKey,
                Attribute &attrVal) {
               self.getContext()
                   ->getOrLoadDialect<annotation::AnnotationDialect>();
               auto annotationOp = self.create<annotation::MarkOp>(ptr);
               annotationOp->setAttr(self.getBuilder().getStringAttr(attrKey),
                                     attrVal);
             })
        .def("create_extract_scalar",
             [](TritonOpBuilder &self, Value &src,
                std::vector<Value> &indices) -> Value {
               llvm::SmallVector<Value> arg_indices;
               for (const auto &i : indices) {
                 if (!i.getType().isIndex()) {
                   arg_indices.push_back(self.create<arith::IndexCastOp>(
                       self.getBuilder().getIndexType(), i));
                 } else {
                   arg_indices.push_back(i);
                 }
               }
               return self.create<tensor::ExtractOp>(src, arg_indices);
             })
        .def("create_index_select_simd",
             [](TritonOpBuilder &self, Value &src, Value &index, int32_t dim,
                std::vector<Value> &srcShape, std::vector<Value> &srcOffset,
                std::vector<int32_t> &readShape,
                std::vector<int32_t> &returnShape) -> Value {
               auto &builder = self.getBuilder();
               auto loc = self.getLastLoc();
               Type elemType;
               if (auto ptrTy = dyn_cast<triton::PointerType>(src.getType())) {
                 elemType = ptrTy.getPointeeType();
               } else {
                 llvm::report_fatal_error(
                     "index_select_simd: src must be pointer type");
               }
               llvm::SmallVector<int64_t> retShape;
               for (const auto &s : returnShape)
                 retShape.push_back(s);
               auto retTensorType = RankedTensorType::get(retShape, elemType);
               llvm::SmallVector<Value> srcShapeIndex;
               for (auto val : srcShape) {
                 if (!val.getType().isIndex())
                   val = self.create<arith::IndexCastOp>(builder.getIndexType(),
                                                         val);
                 srcShapeIndex.push_back(val);
               }
               llvm::SmallVector<Value> srcOffsetIndex;
               for (auto val : srcOffset) {
                 if (!val.getType().isIndex())
                   val = self.create<arith::IndexCastOp>(builder.getIndexType(),
                                                         val);
                 srcOffsetIndex.push_back(val);
               }
               auto op = builder.create<triton::dicp::IndexSelectSimdOp>(
                   loc, retTensorType, src, index,
                   builder.getI32IntegerAttr(dim), srcShapeIndex,
                   srcOffsetIndex, builder.getDenseI32ArrayAttr(readShape));
               return op.getResult();
             })
        .def("create_index_put",
             [](TritonOpBuilder &self, Value &ptr, Value &index, Value &value,
                const int32_t dim, const int64_t indexBoundary,
                std::vector<Value> &endOffset, std::vector<Value> &startOffset,
                std::vector<Value> &dstStride) -> void {
               auto dim_val = self.create<arith::ConstantIntOp>(
                   self.getBuilder().getI32Type(), static_cast<int64_t>(dim));
               auto bound_val = self.create<arith::ConstantIntOp>(
                   self.getBuilder().getI64Type(),
                   static_cast<int64_t>(indexBoundary));
               self.create<triton::dicp::IndexPutOp>(ptr, index, value, dim_val,
                                                     bound_val, endOffset,
                                                     startOffset, dstStride);
             })
        .def("create_gather_out_to_ub",
             [](TritonOpBuilder &self, Value &src, Value &index,
                const int64_t indexBoundary, const int32_t dim,
                std::vector<Value> &srcStride, std::vector<Value> &endOffset,
                std::vector<Value> &startOffset,
                std::optional<Value> &other) -> Value {
               auto elemTy = cast<PointerType>(src.getType()).getPointeeType();
               auto idxShape =
                   cast<RankedTensorType>(index.getType()).getShape();
               std::vector<int64_t> retShape(idxShape.begin(), idxShape.end());
               auto resType = RankedTensorType::get(retShape, elemTy);
               auto bound_val = self.create<arith::ConstantIntOp>(
                   self.getBuilder().getI64Type(),
                   static_cast<int64_t>(indexBoundary));
               auto dim_val = self.create<arith::ConstantIntOp>(
                   self.getBuilder().getI32Type(), static_cast<int64_t>(dim));
               return self.create<triton::dicp::GatherOutToUbOp>(
                   resType, src, index, bound_val, dim_val, srcStride,
                   endOffset, startOffset, other.value_or(Value()));
             })
        .def("create_scatter_ub_to_out",
             [](TritonOpBuilder &self, Value &ptr, Value &value, Value &index,
                const int64_t indexBoundary, const int32_t dim,
                std::vector<Value> &dstStride, std::vector<Value> &endOffset,
                std::vector<Value> &startOffset) -> void {
               auto bound_val = self.create<arith::ConstantIntOp>(
                   self.getBuilder().getI64Type(),
                   static_cast<int64_t>(indexBoundary));
               auto dim_val = self.create<arith::ConstantIntOp>(
                   self.getBuilder().getI32Type(), static_cast<int64_t>(dim));
               self.create<triton::dicp::ScatterUbToOutOp>(
                   ptr, value, index, bound_val, dim_val, dstStride, endOffset,
                   startOffset);
             })
        .def("create_sort",
             [](TritonOpBuilder &self, Value src, int64_t dim,
                bool descending) -> Value {
               auto &builder = self.getBuilder();
               auto op = builder.create<triton::dicp::SortOp>(
                   self.getLastLoc(), src, builder.getI64IntegerAttr(dim),
                   builder.getBoolAttr(descending));
               return op->getResult(0);
             })
        .def("create_flip",
             [](TritonOpBuilder &self, Value src, int64_t dim) -> Value {
               auto op = self.getBuilder().create<triton::dicp::FlipOp>(
                   self.getLastLoc(), src,
                   self.getBuilder().getI64IntegerAttr(dim));
               return op->getResult(0);
             });

    // --- buffer_builder class ---
    struct BufferOpBuilder : public TritonOpBuilder {};

    m.def("load_buffer_dialects", [](MLIRContext &context) {
      DialectRegistry registry;
      registry
          .insert<memref::MemRefDialect, bufferization::BufferizationDialect>();
      context.appendDialectRegistry(registry);
      context.loadAllAvailableDialects();
    });

    py::class_<BufferOpBuilder, TritonOpBuilder>(
        m, "buffer_builder", py::module_local(), py::dynamic_attr())
        .def(py::init<MLIRContext *>())
        .def("get_null_attr",
             [](BufferOpBuilder &self) -> Attribute { return Attribute(); })
        .def("get_str_array_attr",
             [](BufferOpBuilder &self,
                const std::vector<std::string> &array) -> ArrayAttr {
               auto strRefVec = to_vector(llvm::map_range(
                   array, [](const auto &s) { return llvm::StringRef(s); }));
               return self.getBuilder().getStrArrayAttr(
                   llvm::ArrayRef<StringRef>{strRefVec});
             })
        .def("alloc",
             [](BufferOpBuilder &self, Type memrefType) -> Value {
               return self.create<memref::AllocOp>(
                   mlir::cast<MemRefType>(memrefType));
             })
        .def("to_buffer",
             [](BufferOpBuilder &self, Value &src,
                const Attribute &addressSpace) -> Value {
               auto tensorType = dyn_cast<RankedTensorType>(src.getType());
               if (!tensorType) {
                 llvm::report_fatal_error("to_buffer: src must be tensor type");
               }
               auto memrefType = MemRefType::get(tensorType.getShape(),
                                                 tensorType.getElementType(),
                                                 MemRefLayoutAttrInterface{});
               Operation *memref =
                   self.create<bufferization::ToBufferOp>(memrefType, src);
               if (addressSpace) {
                 memref = self.create<memref::MemorySpaceCastOp>(
                     MemRefType::get(memrefType.getShape(),
                                     memrefType.getElementType(),
                                     memrefType.getLayout(), addressSpace),
                     memref->getResult(0));
               }
               return memref->getResult(0);
             })
        .def("to_tensor",
             [](BufferOpBuilder &self, Value &src, bool writable) -> Value {
               const auto &memrefType = mlir::cast<MemRefType>(src.getType());
               auto tensorType = mlir::RankedTensorType::get(
                   memrefType.getShape(), memrefType.getElementType());
               auto hasAddressSpace = memrefType.getMemorySpace();
               if (hasAddressSpace) {
                 MemRefType targetType = MemRefType::get(
                     memrefType.getShape(), memrefType.getElementType(),
                     memrefType.getLayout());
                 return self.create<bufferization::ToTensorOp>(
                     tensorType,
                     self.create<memref::MemorySpaceCastOp>(targetType, src),
                     mlir::UnitAttr::get(self.getContext()),
                     writable ? mlir::UnitAttr::get(self.getContext())
                              : nullptr);
               }
               return self.create<bufferization::ToTensorOp>(
                   tensorType, src, mlir::UnitAttr::get(self.getContext()),
                   writable ? mlir::UnitAttr::get(self.getContext()) : nullptr);
             })
        .def("subview",
             [](BufferOpBuilder &self, Value source,
                std::vector<Value> &offsets, const std::vector<int64_t> &sizes,
                const std::vector<int64_t> &strides) -> Value {
               SmallVector<mlir::OpFoldResult> mixedOffsets;
               auto *context = self.getBuilder().getContext();
               auto &builder = self.getBuilder();
               auto sourceType = mlir::cast<MemRefType>(source.getType());
               int64_t rank = sourceType.getRank();
               if (offsets.size() != rank || sizes.size() != rank ||
                   strides.size() != rank) {
                 throw std::runtime_error(
                     "Number of offsets, sizes, and strides "
                     "must match memref rank");
               }
               for (const auto &offset : offsets) {
                 auto indexType = builder.getIndexType();
                 if (offset.getType() != indexType) {
                   Value offset_val =
                       self.create<arith::IndexCastOp>(indexType, offset);
                   mixedOffsets.push_back(offset_val);
                 } else {
                   mixedOffsets.push_back(offset);
                 }
               }
               constexpr unsigned kIntegerAttrBitWidth = 64;
               SmallVector<mlir::OpFoldResult> mixedSizes;
               SmallVector<mlir::OpFoldResult> mixedStrides;
               for (int64_t i = 0; i < rank; ++i) {
                 int64_t size = sizes[i];
                 int64_t stride = strides[i];
                 int64_t srcDim = sourceType.getDimSize(i);
                 if (size <= 0) {
                   throw std::runtime_error("Expected sizes to be positive");
                 }
                 if (stride <= 0) {
                   throw std::runtime_error("Expected strides to be positive");
                 }
                 if (!ShapedType::isDynamic(srcDim)) {
                   if (size > srcDim) {
                     throw std::runtime_error(
                         "Subview size cannot exceed source dimension size");
                   }
                   if (stride > srcDim) {
                     throw std::runtime_error(
                         "Stride cannot exceed source dimension size");
                   }
                 }
                 mixedSizes.push_back(IntegerAttr::get(
                     IntegerType::get(context, kIntegerAttrBitWidth), size));
                 mixedStrides.push_back(IntegerAttr::get(
                     IntegerType::get(context, kIntegerAttrBitWidth), stride));
               }
               return self.create<memref::SubViewOp>(source, mixedOffsets,
                                                     mixedSizes, mixedStrides);
             });
  }
}

// =============================================================================
// Pass pipeline bindings (from triton_dicp_triton.cc original)
// =============================================================================

void init_triton_dicp_passes_commonir(py::module &&m) {
  m.def("add_vectorize_parallel_loop", [](mlir::PassManager &pm) {
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::dicp::CommonIR::createVectorizeParallelLoopPass());
  });
  m.def("add_annotate_kernel_attrs", [](mlir::PassManager &pm) {
    pm.addPass(mlir::dicp::CommonIR::createAnnotateKernelAttrsPass());
  });
}

void init_triton_dicp_passes_ttir(py::module &&m) {
  m.def("add_auto_blockify", [](mlir::PassManager &pm, int autoBlockifySize) {
    AutoBlockifyOptions opts;
    opts.autoBlockifySize = autoBlockifySize;
    pm.addPass(mlir::triton::createAutoBlockifyPass(opts));
  });

  m.def("add_ascend_legalize", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createAscendLegalizePass());
  });

  m.def("add_triton_to_structure",
        [](mlir::PassManager &pm, bool enableMaskFallbackConversion,
           bool optimizeDynamicOffset) {
          pm.addPass(mlir::triton::createTritonToStructuredPass(
              enableMaskFallbackConversion, optimizeDynamicOffset));
        });

  m.def("add_discrete_mask_access_conversion", [](mlir::PassManager &pm,
                                                  bool compileOn91095,
                                                  bool forceSimtTemplate,
                                                  bool enableSyncBlockLock) {
    DiscreteMaskAccessConversionOptions opts;
    opts.compileOn91095 = compileOn91095;
    opts.forceSimtTemplate = forceSimtTemplate;
    opts.enableSyncBlockLock = enableSyncBlockLock;
    pm.addPass(mlir::triton::createDiscreteMaskAccessConversionPass(opts));
  });

  m.def("add_triton_to_annotation", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createTritonToAnnotationPass());
  });

  m.def("add_triton_to_unstructure",
        [](mlir::PassManager &pm, bool compileOn91095, bool forceSimtTemplate) {
          TritonToUnstructureOptions opts;
          opts.compileOn91095 = compileOn91095;
          opts.forceSimtTemplate = forceSimtTemplate;
          pm.addPass(mlir::triton::createTritonToUnstructurePass(opts));
        });

  m.def("add_triton_to_hivm", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createTritonToHIVMPass());
  });

  m.def("add_triton_to_hfusion", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createTritonToHFusionPass());
  });

  m.def("add_triton_to_llvm", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createTritonToLLVMPass());
  });

  m.def("add_bubble_up_operation", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createBubbleUpOperationPass());
  });

  m.def("add_triton_to_linalg",
        [](mlir::PassManager &pm, bool globalKernel, bool namedOps,
           bool enableNd2nzOnVector, bool enableSelectAnalysis,
           bool compileOn91095) {
          pm.addPass(mlir::triton::createTritonToLinalgPass(
              globalKernel, namedOps, enableNd2nzOnVector, enableSelectAnalysis,
              compileOn91095));
        });

  m.def("add_ascend_npu_ir_legalize",
        [](mlir::PassManager &pm, bool unsafeMode) {
          AscendNPUIRLegalizeOptions opts;
          opts.unsafeMode = unsafeMode;
          pm.addPass(mlir::triton::createAscendNPUIRLegalizePass(opts));
        });

  m.def("add_dynamic_cv_pipeline",
        [](mlir::PassManager &pm, bool compileOn91095) {
          AddDynamicCVPipelineOptions opts;
          opts.compileOn91095 = compileOn91095;
          pm.addPass(mlir::triton::createAddDynamicCVPipelinePass(opts));
        });

  m.def("add_dag_sync", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createDAGSyncPass());
  });

  m.def("add_dag_scope", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createDAGScopePass());
  });

  m.def("add_dag_ssbuffer", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createDAGSSBufferPass());
  });
}

// =============================================================================
// Top-level init: init_triton_dicp_triton
// =============================================================================

void init_triton_dicp_triton(py::module &&m) {
  m.doc() = "Python bindings to the DICP Triton backend (Ascend NPU)";

  auto passes = m.def_submodule("passes");
  init_triton_dicp_passes_commonir(passes.def_submodule("commonir"));
  init_triton_dicp_passes_ttir(passes.def_submodule("ttir"));

  // DICP NPU IR builder, affine types, hivm enums
  init_dicp_ir(m.def_submodule("ir"));

  // Load core dialects
  m.def("load_dialects", [](MLIRContext &context) {
    DialectRegistry registry;
    registry
        .insert<tensor::TensorDialect, memref::MemRefDialect,
                bufferization::BufferizationDialect, arith::ArithDialect,
                cf::ControlFlowDialect, func::FuncDialect,
                linalg::LinalgDialect, index::IndexDialect, math::MathDialect,
                scf::SCFDialect, triton::TritonDialect, affine::AffineDialect,
                LLVM::LLVMDialect, triton::dicp::TritonDicpDialect>();
    mlir::func::registerInlinerExtension(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });
}
