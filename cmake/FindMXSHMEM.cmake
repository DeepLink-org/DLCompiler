set(MXSHMEM_PATH "")

# Add user defined MXSHMEM_HOME to MXSHMEM_PATH
if(DEFINED ENV{MXSHMEM_HOME})
  list(APPEND MXSHMEM_PATH "$ENV{MXSHMEM_HOME}")
endif()

# Add user defined MACA_PATH to MXSHMEM_PATH
if(DEFINED ENV{MACA_PATH})
    list(APPEND MXSHMEM_PATH "$ENV{MACA_PATH}/")
else()
    list(APPEND MXSHMEM_PATH "/opt/maca/")
endif()

message(STATUS "MXSHMEM search path: ${MXSHMEM_PATH}")

find_path(MXSHMEM_INCLUDE_DIR
    NAMES mxshmem.h
    HINTS ${MXSHMEM_PATH}
    PATH_SUFFIXES include include/mxshmem src/include
    NO_DEFAULT_PATH
)

find_library(MXSHMEM_HOST_LIBRARY
    NAMES mxshmem_host
    HINTS ${MXSHMEM_PATH}
    PATH_SUFFIXES lib build/src
    NO_DEFAULT_PATH
)

find_library(MXSHMEM_DEVICE_LIBRARY
    NAMES mxshmem_device
    HINTS ${MXSHMEM_PATH}
    PATH_SUFFIXES lib build/src
    NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MXSHMEM
    REQUIRED_VARS MXSHMEM_HOST_LIBRARY MXSHMEM_DEVICE_LIBRARY MXSHMEM_INCLUDE_DIR
    FAIL_MESSAGE "MXSHMEM not Found, please set 'MXSHMEM_HOME' to mxshmem source code directory OR set 'MACA_PATH' which has mxshmem under it"
)

if(MXSHMEM_FOUND)
    message(STATUS "Found MXSHMEM lib: ${MXSHMEM_HOST_LIBRARY} ${MXSHMEM_DEVICE_LIBRARY}")
    set(MXSHMEM_INCLUDE_DIRS ${MXSHMEM_INCLUDE_DIR})

    if(NOT TARGET MXSHMEM::mxshmem_host)
        add_library(MXSHMEM::mxshmem_host SHARED IMPORTED)
        set_target_properties(MXSHMEM::mxshmem_host PROPERTIES
            IMPORTED_LOCATION "${MXSHMEM_HOST_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${MXSHMEM_INCLUDE_DIR}"
        )
    endif()

    if(NOT TARGET MXSHMEM::mxshmem_device)
        add_library(MXSHMEM::mxshmem_device STATIC IMPORTED)
        set_target_properties(MXSHMEM::mxshmem_device PROPERTIES
            IMPORTED_LOCATION "${MXSHMEM_DEVICE_LIBRARY}"
        )
    endif()
endif()
