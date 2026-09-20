add_subdirectory(Dialect/IR)
if(NOT DEFINED WAFER_TLE_BUILD_CONVERSIONS OR WAFER_TLE_BUILD_CONVERSIONS)
  add_subdirectory(Conversion/DsaToCore)
endif()
