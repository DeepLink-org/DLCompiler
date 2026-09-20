# Common (UnifiedHardware) is already built by FlagTree as target "Common"
# when wafer is loaded as a plugin via TRITON_PLUGIN_DIRS.
if(NOT WAFER_USE_EXTERNAL_TLE)
  add_subdirectory(Common)
endif()
add_subdirectory(Analysis)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Registrar)
