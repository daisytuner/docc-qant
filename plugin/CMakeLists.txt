

set(SOURCE_FILES_PRIVATE
    src/plugin.cpp
    src/qant/blas/gemm.cpp
    src/qant/passes/reduce_quantization_pass.cpp
    src/qant/passes/remapping_pass.cpp
    src/qant/tensor/matmul.cpp
    src/qant/dataflow/library_nodes/math/tensor/matmul_node.cpp
)

add_library(docc-qant-plugin STATIC
    ${SOURCE_FILES_PRIVATE}
)

add_library(docc::qant-plugin ALIAS docc-qant-plugin)
target_include_directories(docc-qant-plugin
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Sanitizer option
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
if(DOCC_QUANT_ENABLE_SANITIZER)
    target_compile_options(docc-qant-plugin PUBLIC ${_SAN_COMPILE_OPTS})
    target_link_options(docc-qant-plugin PUBLIC ${_SAN_LINK_OPTS})
endif()

# Coverage option
option(DOCC_QUANT_ENABLE_COVERAGE "Enable coverage flags" OFF)
if(DOCC_QUANT_ENABLE_COVERAGE)
    target_compile_options(docc-qant-plugin PUBLIC --coverage)
    target_link_options(docc-qant-plugin PUBLIC --coverage)
endif()

target_link_libraries(docc-qant-plugin
    PUBLIC
        sdfg::sdfg
        sdfg::opt
)
set_target_properties(docc-qant-plugin PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

install(TARGETS docc-qant-plugin
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT docc-qant-plugin
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT docc-qant-plugin-development
)

add_subdirectory(tests)
