# set(LLVM_TARGET_DEFINITIONS Passes.td)
# mlir_tablegen(Passes.h.inc -gen-pass-decls --name LinkedToHIVM)
# add_public_tablegen_target(LinkedToHIVMConversionPassIncGen)


# # add_bishengir_dialect(AnnotationOps annotation)
# # add_bishengir_doc(AnnotationOps AnnotationOps Dialects/ -gen-dialect-doc -dialect annotation)

# set(LLVM_TARGET_DEFINITIONS AnnotationOps.td)
# mlir_tablegen(AnnotationOps.h.inc -gen-op-decls)
# # 添加：生成操作定义实现文件
# mlir_tablegen(AnnotationOps.cpp.inc -gen-op-defs)

# mlir_tablegen(AnnotationOpsDialect.h.inc -gen-dialect-decls -dialect=annotation)
# mlir_tablegen(AnnotationOpsDialect.cpp.inc -gen-dialect-defs -dialect=annotation)

# # 修正：添加正确的public tablegen目标
# add_public_tablegen_target(AnnotationOpsIncGen)
