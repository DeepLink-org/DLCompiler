#ifndef TRITON_TO_UNSTRUCTURE_CONVERSION_PASSES
#define TRITON_TO_UNSTRUCTURE_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToUnstructure : Pass<"triton-to-unstructure", "mlir::ModuleOp"> {
    let summary = "Convert Triton for unstructure case";
    let constructor = "triton::createTritonToUnstructurePass()";
    let options = [
        Option<"forceScalarizeMode", "force-scalarize-mode", "bool", "false", 
           "Scalarize unstructured memory access even if structured dimensions are mixed.">
    ];
}

def BubbleUpOperation : Pass<"bubble-up-operation", "mlir::ModuleOp"> {
    let summary = "Propagates tensor extraction operations upward through producers.";
    let description = [{
        Rewrites `tensor.extract` and `tensor.extract_slice` by moving them above producer operations (arithmetic, shape transforms, and Triton-specific ops). By extracting from operands rather than results, the pass decomposes large tensor computations into localized operations on smaller tiles or slices. 

        Key benefits:
        * Reduces intermediate tensor materialization.
        * Lowers memory bandwidth requirements and register pressure.
        
        Operation Modes:
        * Non-aggressive: Only transforms single-use producers to preserve computational sharing.
        * Aggressive: Allows duplicating computations to prioritize memory traffic reduction.
    }];
    let constructor = "triton::createBubbleUpOperationPass()";
    let options = [
        Option<"enableAggressiveMode", "enable-aggressive-mode", "bool", "true", 
           "Enable aggressive bubble up operation.">,
    ];
}

#endif // TRITON_TO_UNSTRUCTURE_CONVERSION_PASSES
