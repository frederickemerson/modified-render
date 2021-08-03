#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "nvcomp::nvcomp" for configuration "Release"
set_property(TARGET nvcomp::nvcomp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(nvcomp::nvcomp PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CUDA;CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/nvcomp.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS nvcomp::nvcomp )
list(APPEND _IMPORT_CHECK_FILES_FOR_nvcomp::nvcomp "${_IMPORT_PREFIX}/lib/nvcomp.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
