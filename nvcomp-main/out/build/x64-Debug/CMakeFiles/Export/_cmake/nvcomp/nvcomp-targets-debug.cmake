#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "nvcomp::nvcomp" for configuration "Debug"
set_property(TARGET nvcomp::nvcomp APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(nvcomp::nvcomp PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CUDA;CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/nvcomp.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS nvcomp::nvcomp )
list(APPEND _IMPORT_CHECK_FILES_FOR_nvcomp::nvcomp "${_IMPORT_PREFIX}/lib/nvcomp.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
