
set(GENERATED_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/target)
file(MAKE_DIRECTORY ${GENERATED_SRC_DIR})

set(COMMONIR_SOURCE_FILES
  codegen_commonir.cc
  codegen_commonir.h
  rt_mod_commonir.cc
)

set(GENERATED_SRCS "")
foreach(file_name IN LISTS COMMONIR_SOURCE_FILES)
  set(src_path ${CMAKE_CURRENT_LIST_DIR}/../commonir/src/target/${file_name})
  set(dst_path ${GENERATED_SRC_DIR}/${file_name})
  
  add_custom_command(
    OUTPUT ${dst_path}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src_path} ${dst_path}
    DEPENDS ${src_path}
    COMMENT "Generating ${file_name} from CommonIR"
    VERBATIM
  )
  list(APPEND GENERATED_SRCS ${dst_path})
endforeach()

set(TILE_LANG_COMMONIR_SRCS
  ${GENERATED_SRCS}  
#   ${CMAKE_CURRENT_SOURCE_DIR}/src/target/codegen_commonir.cc
#   ${CMAKE_CURRENT_SOURCE_DIR}/src/target/rt_mod_commonir.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_COMMONIR_SRCS})