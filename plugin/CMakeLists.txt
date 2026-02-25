

set(SOURCE_FILES_PRIVATE
    src/plugin.cpp
)

add_library(docc-qant-plugin SHARED
    ${SOURCE_FILES_PRIVATE}
)

add_library(docc::qant-plugin ALIAS docc-qant-plugin)

target_include_directories(docc-qant-plugin
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(DOCC_QUANT_ENABLE_SANITIZER)
    target_compile_options(docc-qant-plugin PUBLIC ${_SAN_COMPILE_OPTS})
    target_link_options(docc-qant-plugin PUBLIC ${_SAN_LINK_OPTS})
endif()

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
