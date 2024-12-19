#include "dicp/Dialect/NPU/Transforms/Codegen/NPUCodegen.h"

#define DEBUG_TYPE "npu_codegen"

namespace mlir::dicp::npu {

void NPUCodegen::init(ModuleOp m, const std::string &filename) {
  this->filename = filename;
}

void NPUCodegen::run(ModuleOp s) {
}

void NPUCodegen::store() {
    
}

} // end namespace