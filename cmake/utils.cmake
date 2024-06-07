function(add_dicp_triton_library name)
  cmake_parse_arguments(ARG
    ""
    ""
    "DEPENDS;INCLUDE_DIRS;LINK_LIBS;LINK_COMPONENTS"
    ${ARGN}
    )

  add_library(${name} ${ARG_UNPARSED_ARGUMENTS})

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
  set_property(TARGET ${name} PROPERTY POSITION_INDEPENDENT_CODE ON)
endfunction(add_dicp_triton_library)


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