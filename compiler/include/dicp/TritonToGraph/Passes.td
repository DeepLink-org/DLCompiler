

#ifndef TRITON_TO_CFG_PASSES
#define TRITON_TO_CFG_PASSES

include "mlir/Pass/PassBase.td"

def BuildCFG : Pass<"build-cfg", "mlir::ModuleOp"> {
  let summary = "Build Control Flow Graph from TTIR";
  let description = [{
    This pass analyzes the TTIR and builds a Control Flow Graph (CFG)
    representation. It exports the CFG to multiple formats (text, DOT, JSON, HTML)
    for visualization and analysis.

    The output files will be named as <function_name>_cfg.<ext> and placed in
    the directory specified by the output-dir option.
  }];
  let constructor = "mlir::triton::cfg::createBuildCFGPass()";
  let options = [
    Option<"output-dir", "outputDir", "std::string", "\".\"",
           "Directory where CFG files will be generated">,
  ];
}

#endif // TRITON_TO_CFG_PASSES
