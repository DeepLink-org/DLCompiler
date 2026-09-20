// Frontend-only binding: no FLIR, MK, device lowering, or vendor SDK dependency.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/AllInterfaces.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;
void init_triton_tle(py::module &&m);

void init_triton_wafer(py::module &&m) {
  m.attr("build_role") = "frontend";
  init_triton_tle(m.def_submodule("tle"));
  auto common = m.def_submodule("common");
  common.def("generic_print", [](mlir::ModuleOp mod) {
    std::string text;
    llvm::raw_string_ostream os(text);
    mlir::OpPrintingFlags flags;
    flags.enableDebugInfo();
    flags.printGenericOpForm();
    mod.print(os, flags);
    return text;
  });
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    using namespace mlir;
    DialectRegistry registry;
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect, vector::VectorDialect,
                    func::FuncDialect>();
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    linalg::registerAllDialectInterfaceImplementations(registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
    func::registerAllExtensions(registry);
    scf::registerBufferizableOpInterfaceExternalModels(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });
}
