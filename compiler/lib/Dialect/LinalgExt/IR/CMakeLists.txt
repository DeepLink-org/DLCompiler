add_dicp_triton_library(DICPLinalgExt
    Dialect.cpp
    Ops.cpp 
    
    DEPENDS
    DICPLinalgExtIncGen

    LINK_LIBS PUBLIC
    MLIRLinalgDialect
)