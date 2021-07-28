#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "bitcomp" for configuration "Release"
set_property(TARGET bitcomp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(bitcomp PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CUDA;CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libbitcomp.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS bitcomp )
list(APPEND _IMPORT_CHECK_FILES_FOR_bitcomp "${_IMPORT_PREFIX}/lib/libbitcomp.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
