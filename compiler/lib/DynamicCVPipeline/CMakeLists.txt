file(GLOB PLAN_COMPUTE_BLOCK_PASS_SRC
  "PlanComputeBlock/*.cpp"
)
file(GLOB COMMON_SRC
  "Common/*.cpp"
)

file(GLOB SEPARATE_MEMORY_FROM_COMPUTE_SRC_FILES CONFIGURE_DEPENDS
  "SeparateMemoryFromCompute/*.cpp"
)

file(GLOB ADD_CONTROLFLOW_CONDITION_SRC_FILES CONFIGURE_DEPENDS
    "AddControlFlowCondition/*.cpp"
)

add_triton_library(DynamicCVPipeline
  AddDynamicCVPipeline.cpp
  PlanComputeBlock.cpp
  ${PLAN_COMPUTE_BLOCK_PASS_SRC}
  ${COMMON_SRC}
  SplitDataflow.cpp
  SeparateMemoryFromComputePass.cpp
  AllocMultiCache.cpp
  AddControlFlowCondition.cpp
  ${SEPARATE_MEMORY_FROM_COMPUTE_SRC_FILES}
  ${ADD_CONTROLFLOW_CONDITION_SRC_FILES}

  DEPENDS
  AddDynamicCVPipelinePassIncGen

  LINK_LIBS
  BiShengIRHIVMDialect
  BiShengIRScopeDialect
)