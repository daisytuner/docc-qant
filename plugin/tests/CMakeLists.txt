include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(TEST_FILES
    qant/tensor/conv_test.cpp
    qant/tensor/matmul_test.cpp
    test.cpp
)

# Now simply link against gtest or gtest_main as needed. Eg
add_executable(docc-qant-plugin_test ${TEST_FILES})
target_link_libraries(docc-qant-plugin_test gtest_main docc-qant-plugin)

add_test(NAME docc-qant-plugin_test COMMAND docc-qant-plugin_test)

include(GoogleTest)
gtest_discover_tests(docc-qant-plugin_test)
