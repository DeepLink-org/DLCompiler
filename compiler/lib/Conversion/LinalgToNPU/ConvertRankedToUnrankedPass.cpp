#include "dicp/Conversion/LinalgToNPU/ConvertRankedToUnrankedPass.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
// #include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Pass/PassManager.h"
#include "dicp/Conversion/LinalgToNPU/ConvertRankedToUnrankedPass.hpp"


using namespace mlir;
using namespace mlir::func;

namespace mlir::dicp::npu {

// // 定义 ConvertRankedToUnrankedPass
// struct ConvertRankedToUnrankedPass
//     : public PassWrapper<ConvertRankedToUnrankedPass, OperationPass<ModuleOp>> {



// };


// std::unique_ptr<mlir::Pass> createConvertRankedToUnrankedPass() {
//   return std::make_unique<mlir::dicp::npu::ConvertRankedToUnrankedPass>();
// }


} // namespace mlir::dicp::npu
