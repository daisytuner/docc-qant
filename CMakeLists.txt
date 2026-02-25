cmake_minimum_required(VERSION 3.26)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" DOCC_QUANT_VERSION)
string(STRIP "${DOCC_QUANT_VERSION}" DOCC_QUANT_VERSION)

project(docc-qant VERSION ${DOCC_QUANT_VERSION} DESCRIPTION "Docc plugins for the Q.ANT accelerator")
include(GNUInstallDirs)

#include(cmake/PackageDependencies.cmake)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#option(PACKAGE_WITH_TOOL_DEPS "Include manual tool dependencies in the package metadata (e.g. clang-19, lld-19, which are only called but not linked and thus not covered by automatic detection)" OFF)

set(WITH_DOCC "" CACHE STRING "Path to the docc source directory")

cmake_path(ABSOLUTE_PATH WITH_DOCC BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_VARIABLE WITH_DOCC_ABS)

if(NOT (WITH_DOCC STREQUAL ""))
    # Only add docc if it hasn't already been added by a parent project
    if(NOT TARGET sdfg::sdfg) # WITH_DOCC needs to be set explicitly, do we need this check?
        if(EXISTS "${WITH_DOCC_ABS}/CMakeLists.txt")
            message(STATUS "Using docc source path: ${WITH_DOCC_ABS}")
            set(SDFG_BUILD_TESTS OFF)
            set(BUILD_TESTS OFF)
            set(BUILD_BENCHMARKS OFF)
            set(BUILD_BENCHMARKS_GOOGLE OFF)
            add_subdirectory(${WITH_DOCC_ABS} docc)
        else()
            message(FATAL_ERROR "The provided WITH_DOCC does not contain a valid CMakeLists.txt: ${WITH_DOCC_ABS}")
        endif()
    else()
        message(STATUS "docc already available as target sdfg::sdfg, skipping add_subdirectory")
    endif()
endif()

option(DOCC_QUANT_ENABLE_SANITIZER
       "Enable Address/Memory/Undefined/Leak/Thread sanitizers" OFF)

set(DOCC_QUANT_SANITIZER
    "address"
    CACHE STRING
    "List of sanitizers to enable; supported: \
     address, memory, undefined, leak, thread")

if(DOCC_QUANT_ENABLE_SANITIZER)
  message(STATUS "Enabling sanitizer: ${DOCC_QUANT_SANITIZER}")
  set(_SAN_COMPILE_OPTS
    -fno-omit-frame-pointer
    -fsanitize=${DOCC_QUANT_SANITIZER}
  )
  set(_SAN_LINK_OPTS
    -fsanitize=${DOCC_QUANT_SANITIZER}
  )
endif()

# Coverage option
option(DOCC_QUANT_ENABLE_COVERAGE "Enable coverage flags" OFF)

# libsdfg
if(NOT TARGET sdfg::sdfg)
    message(STATUS "Looking for pre-installed libsdfg... (may not install autotuner fully)")
    find_package(sdfg CONFIG REQUIRED)
endif()

add_subdirectory(plugin)