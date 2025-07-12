#ifndef CONVERT_RANKED_TO_UNRANKED_PASS_H
#define CONVERT_RANKED_TO_UNRANKED_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h" 
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
// #include "llvm/ADT/Optional.h"  // 添加缺失的头文件

#define DICPRTU_DEBUG_TYPE "convert-ranked-to-unranked"

using namespace mlir;
using namespace mlir::func;

namespace mlir {
namespace dicp {
namespace npu {

inline bool isLegalForUnrankedTypeConversion(Operation *op, TypeConverter &typeConverter) {
    llvm::outs() << "Checking legality of operation: ";
    op->getName().print(llvm::outs());
    llvm::outs() << "\n";
    for (Type type : op->getOperandTypes()) {
        if (auto memrefTy = dyn_cast<MemRefType>(type)) {
            if (memrefTy.hasStaticShape()) {
                llvm::outs() << "Operand type has static shape, considered legal: ";
                type.print(llvm::outs());
                llvm::outs() << "\n";
                continue;
            }
        }
        if (!typeConverter.isLegal(type)) {
            llvm::outs() << "Operand type is illegal: ";
            type.print(llvm::outs());
            llvm::outs() << "\n";
            return false;
        }
    }
    for (Type type : op->getResultTypes()) {
        if (auto memrefTy = dyn_cast<MemRefType>(type)) {
            if (memrefTy.hasStaticShape()) {
                llvm::outs() << "Result type has static shape, considered legal: ";
                type.print(llvm::outs());
                llvm::outs() << "\n";
                continue;
            }
        }
        if (!typeConverter.isLegal(type)) {
            llvm::outs() << "Result type is illegal: ";
            type.print(llvm::outs());
            llvm::outs() << "\n";
            return false;
        }
    }
    llvm::outs() << "Operation is legal: ";
    op->getName().print(llvm::outs());
    llvm::outs() << "\n";
    return true;
}

inline Value convertValueIfNeeded(OpBuilder &builder, Location loc, Value value, const TypeConverter &typeConverter) {
    Type originalType = value.getType();
    llvm::outs() << "Converting value with original type: ";
    originalType.print(llvm::outs());
    llvm::outs() << "\n";
    Type convertedType = typeConverter.convertType(originalType);
    if (convertedType == originalType) {
        llvm::outs() << "Type remains the same, no conversion needed.\n";
        return value;
    }
    if (auto memrefTy = dyn_cast<MemRefType>(originalType)) {
        llvm::outs() << "Converting memref type to: ";
        convertedType.print(llvm::outs());
        llvm::outs() << "\n";
        return builder.create<memref::CastOp>(loc, convertedType, value);
    }
    llvm::outs() << "No suitable conversion for type, returning original value.\n";
    return value;
}

struct ScfForOpTypeConversionPattern : public OpConversionPattern<scf::ForOp> {
    using OpConversionPattern<scf::ForOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        llvm::outs() << "Processing scf.for operation: ";
        op->print(llvm::outs());
        llvm::outs() << "\n";
        SmallVector<Type> newResultTypes;
        if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), newResultTypes))) {
            llvm::outs() << "Failed to convert result types for scf.for operation.\n";
            return failure();
        }
        llvm::outs() << "Converted result types for scf.for operation: ";
        for (Type type : newResultTypes) {
            type.print(llvm::outs());
            llvm::outs() << " ";
        }
        llvm::outs() << "\n";

        SmallVector<Value> newInitArgs;
        for (Value initArg : adaptor.getInitArgs()) {
            newInitArgs.push_back(convertValueIfNeeded(rewriter, op.getLoc(), initArg, *getTypeConverter()));
        }

        auto newOp = rewriter.create<scf::ForOp>(
            op.getLoc(),
            adaptor.getLowerBound(),
            adaptor.getUpperBound(),
            adaptor.getStep(),
            newInitArgs,
            [&](OpBuilder &builder, Location loc, Value iv, ValueRange args) {
                llvm::outs() << "Inside scf.for loop body, induction variable: ";
                iv.print(llvm::outs());
                llvm::outs() << "\n";
                Block &oldBlock = op.getRegion().front();
                llvm::DenseMap<Value, Value> valueMap;
                valueMap[oldBlock.getArgument(0)] = iv;
                auto newArgsIter = args.begin();
                for (auto oldArg : oldBlock.getArguments().drop_front()) {
                    valueMap[oldArg] = *newArgsIter;
                    ++newArgsIter;
                }

                for (auto &oldOp : oldBlock.without_terminator()) {
                    SmallVector<Value> newOperands;
                    for (Value operand : oldOp.getOperands()) {
                        if (valueMap.count(operand)) {
                            newOperands.push_back(valueMap[operand]);
                        } else {
                            newOperands.push_back(operand);
                        }
                    }
                    Operation *newOp = builder.clone(oldOp);
                    newOp->setOperands(newOperands);
                    for (auto resultPair : llvm::zip(oldOp.getResults(), newOp->getResults())) {
                        valueMap[std::get<0>(resultPair)] = std::get<1>(resultPair);
                    }
                }

                auto yieldOp = cast<scf::YieldOp>(oldBlock.getTerminator());
                SmallVector<Value> newYieldOperands;
                for (auto operand : yieldOp.getOperands()) {
                    if (valueMap.count(operand)) {
                        newYieldOperands.push_back(valueMap[operand]);
                    } else {
                        newYieldOperands.push_back(operand);
                    }
                }
                builder.create<scf::YieldOp>(loc, newYieldOperands);
            }
        );

        rewriter.replaceOp(op, newOp.getResults());
        llvm::outs() << "Replaced scf.for operation with new one.\n";
        return success();
    }
};

struct CallOpTypeConversionPattern : public OpConversionPattern<CallOp> {
    using OpConversionPattern<CallOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(CallOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        llvm::outs() << "Processing CallOp: ";
        op->print(llvm::outs());
        llvm::outs() << "\n";
        SmallVector<Type> newResultTypes;
        if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), newResultTypes))) {
            llvm::outs() << "Failed to convert result types for CallOp.\n";
            return failure();
        }
        llvm::outs() << "Converted result types for CallOp: ";
        for (Type type : newResultTypes) {
            type.print(llvm::outs());
            llvm::outs() << " ";
        }
        llvm::outs() << "\n";

        SmallVector<Value> newOperands;
        for (Value operand : adaptor.getOperands()) {
            newOperands.push_back(convertValueIfNeeded(rewriter, op.getLoc(), operand, *getTypeConverter()));
        }

        auto newOp = rewriter.create<CallOp>(op.getLoc(), op.getCallee(), newResultTypes, newOperands);
        rewriter.replaceOp(op, newOp.getResults());
        llvm::outs() << "Replaced CallOp with new one.\n";
        return success();
    }
};

struct ReturnOpTypeConversionPattern : public OpConversionPattern<ReturnOp> {
    using OpConversionPattern<ReturnOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        llvm::outs() << "Processing ReturnOp: ";
        op->print(llvm::outs());
        llvm::outs() << "\n";
        SmallVector<Value> newOperands;
        for (Value operand : adaptor.getOperands()) {
            newOperands.push_back(convertValueIfNeeded(rewriter, op.getLoc(), operand, *getTypeConverter()));
        }
        rewriter.replaceOpWithNewOp<ReturnOp>(op, newOperands);
        llvm::outs() << "Replaced ReturnOp with new one.\n";
        return success();
    }
};

inline void populateTypeConversionPatterns1(RewritePatternSet &patterns, TypeConverter &typeConverter, ConversionTarget &target) {
    populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns, typeConverter);
    target.addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp func) { 
            bool isLegal = typeConverter.isSignatureLegal(func.getFunctionType());
            llvm::outs() << "FuncOp " << func.getName() << " signature is " << (isLegal ? "legal" : "illegal") << "\n";
            return isLegal; 
        });

    patterns.add<CallOpTypeConversionPattern>(typeConverter, patterns.getContext());
    target.addDynamicallyLegalOp<CallOp>(
        [&](CallOp op) { 
            bool isLegal = true;
            for (Type type : op.getOperandTypes()) {
                if (auto memrefTy = dyn_cast<MemRefType>(type)) {
                    if (memrefTy.hasStaticShape()) continue;
                }
                if (!typeConverter.isLegal(type)) {
                    isLegal = false;
                    break;
                }
            }
            for (Type type : op.getResultTypes()) {
                if (auto memrefTy = dyn_cast<MemRefType>(type)) {
                    if (memrefTy.hasStaticShape()) continue;
                }
                if (!typeConverter.isLegal(type)) {
                    isLegal = false;
                    break;
                }
            }
            llvm::outs() << "CallOp " << op.getCallee() << " is " << (isLegal ? "legal" : "illegal") << "\n";
            return isLegal; 
        });

    patterns.add<ReturnOpTypeConversionPattern>(typeConverter, patterns.getContext());
    target.addDynamicallyLegalOp<ReturnOp>(
        [&](ReturnOp op) { 
            bool isLegal = true;
            for (Type type : op.getOperandTypes()) {
                if (auto memrefTy = dyn_cast<MemRefType>(type)) {
                    if (memrefTy.hasStaticShape()) continue;
                }
                if (!typeConverter.isLegal(type)) {
                    isLegal = false;
                    break;
                }
            }
            llvm::outs() << "ReturnOp is " << (isLegal ? "legal" : "illegal") << "\n";
            return isLegal; 
        });

    patterns.add<ScfForOpTypeConversionPattern>(typeConverter, patterns.getContext());
    target.addDynamicallyLegalOp<scf::ForOp>(
        [&](scf::ForOp op) { 
            bool isLegal = isLegalForUnrankedTypeConversion(op, typeConverter);
            llvm::outs() << "scf.for operation is " << (isLegal ? "legal" : "illegal") << "\n";
            return isLegal; 
        });
}

struct UnrealizedConversionCastPattern : public OpConversionPattern<UnrealizedConversionCastOp> {
    using OpConversionPattern<UnrealizedConversionCastOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (op.getInputs().size() != 1 || op.getOutputs().size() != 1) {
            return failure();
        }

        Value input = op.getInputs()[0];
        Type resultType = op.getOutputs()[0].getType();

        if (isa<MemRefType>(input.getType()) && isa<MemRefType>(resultType)) {
            rewriter.replaceOpWithNewOp<memref::CastOp>(op, resultType, input);
            return success();
        }
        
        rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, resultType, input);
        return success();
    }
};

struct ConvertRankedToUnrankedPass : public PassWrapper<ConvertRankedToUnrankedPass, OperationPass<ModuleOp>> {
    
    void runOnOperation() override {
        llvm::outs() << "Starting ConvertRankedToUnrankedPass on module: ";
        getOperation()->print(llvm::outs());
        llvm::outs() << "\n";
        auto module = getOperation();
        MLIRContext *ctx = &getContext();

        TypeConverter typeConverter;

        typeConverter.addConversion([&](Type t) -> Type {
            llvm::outs() << "Checking type for conversion: ";
            t.print(llvm::outs());
            llvm::outs() << "\n";

            // 使用 isa 进行类型检查
            if (isa<MemRefType>(t)) {
                llvm::outs() << "Type is recognized as MemRefType before dyn_cast.\n";
            } else {
                llvm::outs() << "Type is not recognized as MemRefType before dyn_cast.\n";
            }

            if (auto memrefTy = dyn_cast<MemRefType>(t)) {
                llvm::outs() << "MemRef type details: ";
                memrefTy.print(llvm::outs());
                llvm::outs() << "\n";

                bool hasDynamicDim = false;
                for (int64_t dim : memrefTy.getShape()) {
                    if (ShapedType::isDynamic(dim)) {
                            hasDynamicDim = true;
                            break;
                        }
                    }
                
                if (hasDynamicDim) {
                    llvm::outs() << "Converting MemRef with dynamic shape to single-dimensional dynamic shape\n";
                    return MemRefType::get(
                        {ShapedType::kDynamic}, 
                        memrefTy.getElementType(),
                        AffineMap(),
                        memrefTy.getMemorySpace()
                    );
                }
                
                llvm::outs() << "No dynamic dimension found, no conversion needed.\n";
            } else {
                llvm::outs() << "dyn_cast to MemRefType failed for type: ";
                t.print(llvm::outs());
                llvm::outs() << "\n";
            }            
            
            llvm::outs() << "No conversion needed for type.\n";
            return t;
        });

        // 修复材料化函数的返回类型
        typeConverter.addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                                                  ValueRange inputs, Location loc) -> Value {
            llvm::outs() << "Source materialization called. Inputs count: " << inputs.size() << "\n";
            if (inputs.size() != 1) {
                llvm::outs() << "Expected 1 input, got " << inputs.size() << ", returning nullptr.\n";
                return nullptr;
            }
            
            Value input = inputs[0];
            if (isa<MemRefType>(input.getType()) && isa<MemRefType>(resultType)) {
                llvm::outs() << "Creating memref::CastOp for source materialization\n";
                return builder.create<memref::CastOp>(loc, resultType, input);
            }
            
            llvm::outs() << "No suitable cast operation for source materialization\n";
            return nullptr;
        });

        typeConverter.addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                                                  ValueRange inputs, Location loc) -> Value {
            llvm::outs() << "Target materialization called. Inputs count: " << inputs.size() << "\n";
            if (inputs.size() != 1) {
                llvm::outs() << "Expected 1 input, got " << inputs.size() << ", returning nullptr.\n";
                return nullptr;
            }
            
            Value input = inputs[0];
            if (isa<MemRefType>(input.getType()) && isa<MemRefType>(resultType)) {
                llvm::outs() << "Creating memref::CastOp for target materialization\n";
                return builder.create<memref::CastOp>(loc, resultType, input);
            }
            
            llvm::outs() << "No suitable cast operation for target materialization\n";
            return nullptr;
        });

        ConversionTarget target(*ctx);
        target.addLegalDialect<arith::ArithDialect>();
        target.addLegalDialect<func::FuncDialect>();
        target.addLegalDialect<linalg::LinalgDialect>();
        target.addLegalDialect<memref::MemRefDialect>();
        target.addLegalDialect<tensor::TensorDialect>();
        target.addLegalDialect<scf::SCFDialect>(); 
        target.addLegalOp<UnrealizedConversionCastOp>();

        target.markUnknownOpDynamicallyLegal([&](Operation *op) {
            bool isLegal = isLegalForUnrankedTypeConversion(op, typeConverter);
            llvm::outs() << "Operation ";
            op->getName().print(llvm::outs());
            llvm::outs() << " is " << (isLegal ? "legal" : "illegal") << "\n";
            return isLegal;
        });

        RewritePatternSet patterns(ctx);
        populateTypeConversionPatterns1(patterns, typeConverter, target);
        patterns.add<UnrealizedConversionCastPattern>(typeConverter, ctx);

        // 添加标准函数转换模式
        // populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns, typeConverter);
        // populateCallOpTypeConversionPattern(patterns, typeConverter);
        // populateReturnOpTypeConversionPattern(patterns, typeConverter);
        // populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
        // populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, typeConverter);

        if (failed(applyFullConversion(module, target, std::move(patterns)))) {
            llvm::outs() << "Failed to apply full conversion on module.\n";
            signalPassFailure();
        } else {
            llvm::outs() << "Successfully applied full conversion on module.\n";
        }

        // 后处理：清理不必要的cast操作
        module.walk([&](memref::CastOp castOp) {
            if (castOp.getOperand().getType() == castOp.getType()) {
                castOp.replaceAllUsesWith(castOp.getOperand());
                castOp.erase();
                llvm::outs() << "Removed redundant cast operation\n";
            }
        });

        llvm::outs() << "After ConvertRankedToUnrankedPass on module: ";
        getOperation()->print(llvm::outs());
        llvm::outs() << "\nFinished ConvertRankedToUnrankedPass on module.\n";
    }

    StringRef getArgument() const final { return "convert-ranked-to-unranked"; }
    StringRef getDescription() const final {
        return "Convert memref types with dynamic shape to single-dimensional dynamic ones";
    }
};

inline std::unique_ptr<mlir::Pass> createConvertRankedToUnrankedPass() {
    return std::make_unique<ConvertRankedToUnrankedPass>();
}

} // namespace npu
} // namespace dicp
} // namespace mlir

#endif // CONVERT_RANKED_TO_UNRANKED_PASS_H