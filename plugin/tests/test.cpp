#include <gtest/gtest.h>

#include <docc/qant/plugin.h>
#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/serializer/json_serializer.h>

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    sdfg::codegen::register_default_dispatchers();
    sdfg::serializer::register_default_serializers();

    register_docc_plugin();

    return RUN_ALL_TESTS();
}
