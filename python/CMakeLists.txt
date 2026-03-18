find_package(pybind11 CONFIG REQUIRED)

set(PYTHON_SOURCES
    bindings/bindings.cpp
)

pybind11_add_module(_qant ${PYTHON_SOURCES})
set_target_properties(_qant PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
target_include_directories(_qant
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/bindings>
)

target_link_libraries(_qant PRIVATE docc::qant-plugin)

target_include_directories(_qant PRIVATE
    ${CMAKE_SOURCE_DIR}/plugin/include
)

# Install artifacts for the Python wheel
install(TARGETS _qant DESTINATION docc/qant COMPONENT docc_qant)
