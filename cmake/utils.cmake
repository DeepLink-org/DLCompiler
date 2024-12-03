function(add_dicp_object name)
  cmake_parse_arguments(ARG "" "" "DEPENDS;LINK_LIBS" ${ARGN})
  add_library(${name} OBJECT)
  target_sources(${name}
    PRIVATE ${ARG_UNPARSED_ARGUMENTS}
    INTERFACE $<TARGET_OBJECTS:${name}>
  )

  if(ARG_DEPENDS)
    add_dependencies(${name} ${ARG_DEPENDS})
  endif()
  if (ARG_INCLUDE_DIRS)
    target_include_directories(${name} PUBLIC ${ARG_INCLUDE_DIRS})
  endif()

  if(ARG_LINK_LIBS)
    target_link_libraries(${name} PUBLIC ${ARG_LINK_LIBS})
  endif()
endfunction(add_dicp_object)

set_property(GLOBAL PROPERTY POLARIS_LIBS "")
function(add_dicp_triton_library name)
  set_property(GLOBAL APPEND PROPERTY POLARIS_LIBS ${name})
  add_dicp_object(${name} ${ARGN})
  llvm_update_compile_flags(${name})
endfunction()


function(add_dicp_triton_executable name)
  cmake_parse_arguments(ARG
    ""
    ""
    "DEPENDS;INCLUDE_DIRS;LINK_LIBS;DEFINE"
    ${ARGN}
    )

  if (EXCLUDE_FROM_ALL)
    add_executable(${name} EXCLUDE_FROM_ALL ${ARG_UNPARSED_ARGUMENTS})
  else()
    add_executable(${name} ${ARG_UNPARSED_ARGUMENTS})
  endif()

  if (ARG_DEPENDS)
    add_dependencies(${name} ${ARG_DEPENDS})
  endif()

  if (ARG_INCLUDE_DIRS)
    target_include_directories(${name} PUBLIC ${ARG_INCLUDE_DIRS})
  endif()
  target_include_directories(${name} PUBLIC ${DC_TRITON_INCLUDE_DIR})

  if (ARG_LINK_LIBS)
    target_link_libraries(${name} PUBLIC ${ARG_LINK_LIBS})
  endif()
  target_link_libraries(${name} PUBLIC ${DC_TRITON_LINK_DIR})

  if (ARG_DEFINE)
    target_compile_definitions(${name} ${ARG_DEFINE})
  endif()
endfunction(add_dicp_triton_executable)

macro(dicp_add_all_subdirs)
  FILE(GLOB _CHILDREN RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
  SET(_DIRLIST "")
  foreach(_CHILD ${_CHILDREN})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${_CHILD} AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_CHILD}/CMakeLists.txt)
      LIST(APPEND _DIRLIST ${_CHILD})
    endif()
  endforeach()

  foreach(subdir ${_DIRLIST})
    add_subdirectory(${subdir})
  endforeach()
endmacro()

function(add_dicp_compiler_dialect target_prefix FileName DialectName)
  set(LLVM_TARGET_DEFINITIONS ${FileName}Ops.td)
  mlir_tablegen(${FileName}Ops.h.inc -gen-op-decls)
  mlir_tablegen(${FileName}Ops.cpp.inc -gen-op-defs)
  mlir_tablegen(${FileName}Types.h.inc -gen-typedef-decls -typedefs-dialect=${DialectName})
  mlir_tablegen(${FileName}Types.cpp.inc -gen-typedef-defs -typedefs-dialect=${DialectName})
  mlir_tablegen(${FileName}Dialect.h.inc -gen-dialect-decls -dialect=${DialectName})
  mlir_tablegen(${FileName}Dialect.cpp.inc -gen-dialect-defs -dialect=${DialectName})
  add_public_tablegen_target(${target_prefix}IncGen)
  add_dependencies(mlir-headers ${target_prefix}IncGen)
endfunction()

function(add_dicp_compiler_passes target_prefix)
  set(LLVM_TARGET_DEFINITIONS Passes.td)
  mlir_tablegen(Passes.h.inc -gen-pass-decls)
  add_public_tablegen_target(${target_prefix}PassesGen)
  add_dependencies(mlir-headers ${target_prefix}PassesGen)
endfunction()