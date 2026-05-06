#include <gtest/gtest.h>

#include <docc/qant/plugin.h>
#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/plugins/plugins.h>
#include <sdfg/serializer/json_serializer.h>
#include <sdfg/visualizer/dot_visualizer.h>

#include "sdfg_debug_dump.h"
#include "so_compile.h"

std::optional<std::filesystem::path> test_output_dir;

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    sdfg::codegen::register_default_dispatchers();
    sdfg::serializer::register_default_serializers();

    docc::qant::register_plugin(docc::qant::docc_context);

#ifdef DOCC_TESTS_ENABLE_DUMP
    test_output_dir = std::filesystem::current_path() / "test_outputs";
#endif

    return RUN_ALL_TESTS();
}


void dump_sdfg(const sdfg::StructuredSDFG& sdfg, const std::string& step) {
    if (test_output_dir) {
        auto info = ::testing::UnitTest::GetInstance()->current_test_info();
        auto suite_name = info->test_suite_name();
        auto test_name = info->name();
        auto base_path = test_output_dir.value() / suite_name / test_name;
        std::filesystem::create_directories(base_path);
        sdfg::serializer::JSONSerializer::writeToFile(sdfg, base_path / (sdfg.name() + "." + step + ".sdfg.json"));
        sdfg::visualizer::DotVisualizer::writeToFile(sdfg, base_path / (sdfg.name() + "." + step + ".sdfg.dot"));
    }
}
