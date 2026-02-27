cmake_minimum_required(VERSION 3.26)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" DOCC_QUANT_VERSION)
string(STRIP "${DOCC_QUANT_VERSION}" DOCC_QUANT_VERSION)

project(docc-qant VERSION ${DOCC_QUANT_VERSION} DESCRIPTION "Docc plugins for the Q.ANT accelerator")
include(GNUInstallDirs)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add docc dependency
add_subdirectory(3rdParty/docc)

# Add native plugin
add_subdirectory(plugin)

# Add bindings
option(BUILD_QANT_PYTHON_BINDINGS "Build Python bindings" OFF)
if(BUILD_QANT_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()
