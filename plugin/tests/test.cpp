#include <gtest/gtest.h>

#include <docc/qant/plugin.h>
#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/plugins/plugins.h>
#include <sdfg/serializer/json_serializer.h>

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    sdfg::codegen::register_default_dispatchers();
    sdfg::serializer::register_default_serializers();

    sdfg::plugins::Context context = sdfg::plugins::Context::global_context();
    docc::qant::register_plugin(context);

    return RUN_ALL_TESTS();
}
