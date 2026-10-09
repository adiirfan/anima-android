cmake_minimum_required(VERSION 3.22)
project(anima LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
# Qualcomm QAIRT 2.50.40 (QNN headers): -DQNN_SDK from the Gradle build, or QNN_SDK_ROOT
if(NOT QNN_SDK)
  if(DEFINED ENV{QNN_SDK_ROOT})
    set(QNN_SDK $ENV{QNN_SDK_ROOT})
  else()
    message(FATAL_ERROR "Set qnn.sdk.dir in local.properties (or QNN_SDK_ROOT) to the QAIRT 2.50.40 folder")
  endif()
endif()

add_library(anima SHARED jni_bridge.cpp qnn_runner.cpp pipeline.cpp klein.cpp gemma.cpp qimg.cpp)
target_include_directories(anima PRIVATE ${QNN_SDK}/include/QNN ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(anima PRIVATE -O3 -Wno-unused-parameter)
target_link_libraries(anima android log dl)
