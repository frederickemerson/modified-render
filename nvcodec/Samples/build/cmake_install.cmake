# Install script for directory: C:/Users/Jeanette/Documents/hrender/hybrid/hrender/nvcodec/Samples

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX ".")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("C:/Users/Jeanette/Documents/hrender/hybrid/hrender/nvcodec/Samples/build/AppEncode/AppEncD3D12/cmake_install.cmake")
  include("C:/Users/Jeanette/Documents/hrender/hybrid/hrender/nvcodec/Samples/build/AppEncode/AppEncD3D11/cmake_install.cmake")
  include("C:/Users/Jeanette/Documents/hrender/hybrid/hrender/nvcodec/Samples/build/AppEncode/AppEncD3D9/cmake_install.cmake")
  include("C:/Users/Jeanette/Documents/hrender/hybrid/hrender/nvcodec/Samples/build/AppDecode/AppDecD3D/cmake_install.cmake")
  include("C:/Users/Jeanette/Documents/hrender/hybrid/hrender/nvcodec/Samples/build/AppEncode/AppEncCuda/cmake_install.cmake")
  include("C:/Users/Jeanette/Documents/hrender/hybrid/hrender/nvcodec/Samples/build/AppEncode/AppEncDec/cmake_install.cmake")
  include("C:/Users/Jeanette/Documents/hrender/hybrid/hrender/nvcodec/Samples/build/AppEncode/AppEncLowLatency/cmake_install.cmake")
  include("C:/Users/Jeanette/Documents/hrender/hybrid/hrender/nvcodec/Samples/build/AppEncode/AppEncME/cmake_install.cmake")
  include("C:/Users/Jeanette/Documents/hrender/hybrid/hrender/nvcodec/Samples/build/AppEncode/AppEncPerf/cmake_install.cmake")
  include("C:/Users/Jeanette/Documents/hrender/hybrid/hrender/nvcodec/Samples/build/AppEncode/AppEncQual/cmake_install.cmake")
  include("C:/Users/Jeanette/Documents/hrender/hybrid/hrender/nvcodec/Samples/build/AppEncode/AppMotionEstimationVkCuda/cmake_install.cmake")
  include("C:/Users/Jeanette/Documents/hrender/hybrid/hrender/nvcodec/Samples/build/AppTranscode/AppTrans/cmake_install.cmake")
  include("C:/Users/Jeanette/Documents/hrender/hybrid/hrender/nvcodec/Samples/build/AppTranscode/AppTransOneToN/cmake_install.cmake")
  include("C:/Users/Jeanette/Documents/hrender/hybrid/hrender/nvcodec/Samples/build/AppTranscode/AppTransPerf/cmake_install.cmake")
  include("C:/Users/Jeanette/Documents/hrender/hybrid/hrender/nvcodec/Samples/build/AppDecode/AppDec/cmake_install.cmake")
  include("C:/Users/Jeanette/Documents/hrender/hybrid/hrender/nvcodec/Samples/build/AppDecode/AppDecGL/cmake_install.cmake")
  include("C:/Users/Jeanette/Documents/hrender/hybrid/hrender/nvcodec/Samples/build/AppDecode/AppDecImageProvider/cmake_install.cmake")
  include("C:/Users/Jeanette/Documents/hrender/hybrid/hrender/nvcodec/Samples/build/AppDecode/AppDecLowLatency/cmake_install.cmake")
  include("C:/Users/Jeanette/Documents/hrender/hybrid/hrender/nvcodec/Samples/build/AppDecode/AppDecMem/cmake_install.cmake")
  include("C:/Users/Jeanette/Documents/hrender/hybrid/hrender/nvcodec/Samples/build/AppDecode/AppDecMultiFiles/cmake_install.cmake")
  include("C:/Users/Jeanette/Documents/hrender/hybrid/hrender/nvcodec/Samples/build/AppDecode/AppDecMultiInput/cmake_install.cmake")
  include("C:/Users/Jeanette/Documents/hrender/hybrid/hrender/nvcodec/Samples/build/AppDecode/AppDecPerf/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "C:/Users/Jeanette/Documents/hrender/hybrid/hrender/nvcodec/Samples/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
