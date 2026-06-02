cmake_minimum_required(VERSION 3.26)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" DOCC_QUANT_VERSION)
string(STRIP "${DOCC_QUANT_VERSION}" DOCC_QUANT_VERSION)

project(docc-qant VERSION ${DOCC_QUANT_VERSION} DESCRIPTION "Docc plugins for the Q.ANT accelerator")
include(GNUInstallDirs)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add docc dependency
set(WITH_DOCC "" CACHE STRING "Path to the docc source directory")

cmake_path(ABSOLUTE_PATH WITH_DOCC BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_VARIABLE WITH_DOCC_ABS)

if(NOT TARGET docc::core)
    if(NOT (WITH_DOCC STREQUAL ""))
        if(EXISTS "${WITH_DOCC_ABS}/CMakeLists.txt")
            message(STATUS "Using docc from source path: ${WITH_DOCC}")
            set(SDFG_BUILD_TESTS OFF)
            set(BUILD_TESTS OFF)
            set(BUILD_BENCHMARKS OFF)
            set(BUILD_BENCHMARKS_GOOGLE OFF)
            add_subdirectory(${WITH_DOCC_ABS} docc)
        else()
            message(FATAL_ERROR "The provided WITH_DOCC does not contain a valid CMakeLists.txt: ${WITH_DOCC}")
        endif()
    else()
        add_subdirectory(3rdParty/docc)
    endif()
endif()

# Add native plugin
add_subdirectory(plugin)

# Add bindings
option(BUILD_QANT_PYTHON_BINDINGS "Build Python bindings" OFF)
if(BUILD_QANT_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()
