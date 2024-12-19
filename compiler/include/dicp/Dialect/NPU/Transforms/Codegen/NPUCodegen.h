#pragma once

using namespace llvm;
namespace mlir::dicp::npu {
class NPUCodegen {
public:
  NPUCodegen() {}
  void init(ModuleOp m, const std::string &filename);
  void run(ModuleOp m);
  void store();

private:
  std::string filename;
};
}