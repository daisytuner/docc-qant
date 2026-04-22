#include <gtest/gtest.h>

#include <docc/qant/plugin.h>
#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/plugins/plugins.h>
#include <sdfg/serializer/json_serializer.h>

#include "docc/compile/file_compiler.h"
#include "docc/target/docc_target.h"
#include "sdfg/codegen/code_generators/cpp_code_generator.h"
#include "sdfg/visualizer/dot_visualizer.h"

static std::optional<std::filesystem::path> test_output_dir;

namespace docc::qant {
sdfg::plugins::Context docc_context = sdfg::plugins::Context::global_context();
}

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

namespace docc::qant {

static std::unique_ptr<docc::util::DefaultDoccPaths> find_docc_paths() {
    return std::make_unique<
        util::DefaultDoccPaths>(DOCC_PATHS_BIN, DOCC_PATHS_SRC, util::DefaultDoccPaths::DoccRootMode::CMake);
}

static std::shared_ptr<docc::util::DefaultDoccPaths> docc_paths = find_docc_paths();

std::string so_compile(sdfg::StructuredSDFG& sdfg, sdfg::plugins::Context& context) {
    auto dir = test_output_dir ? *test_output_dir : std::filesystem::path("/tmp");
    auto info = ::testing::UnitTest::GetInstance()->current_test_info();
    auto suite_name = info->test_suite_name();
    auto test_name = info->name();
    auto base_path = test_output_dir.value() / suite_name / test_name;
    std::filesystem::create_directories(base_path);


    docc::compile::SrcFileCompilerBuilder compile_builder;
    compile_builder.set_compiler("g++")
        .set_from_paths(docc_paths)
        .set_src_extension("cpp")
        .set_bin_extension("so")
        .set_output_dir(base_path)
        .add_common_option("-fPIC")
        .add_common_option("-O3")
        .add_common_option("-fopenmp")
        .add_common_option("-march=native")
        .add_common_option("-mtune=native")
        .add_compile_option("-funroll-loops")
        .add_link_option("-shared")
        .add_link_option("-ldaisy_rtl")
        .add_link_option("-larg_capture_io")
        .add_link_option("-lm")
        .add_link_option("-lstdc++");

    compile_builder.add_common_option("-g");

    qant_target.apply_additional_compile_options(compile_builder);
    docc::target::add_highway_build_support(compile_builder);

    auto fcomp_handler = compile_builder.build();
    docc::compile::CodegenBuildPool pool(1);

    std::shared_ptr<sdfg::codegen::CodeSnippetFactory> snippet_factory = fcomp_handler->create_snippet_factory(sdfg);
    sdfg::analysis::AnalysisManager ana(sdfg);
    auto instrumentation_plan = sdfg::codegen::InstrumentationPlan::none(sdfg);
    auto arg_capture_plan = sdfg::codegen::ArgCapturePlan::none(sdfg);
    sdfg::codegen::CPPCodeGenerator generator(sdfg, ana, *instrumentation_plan, *arg_capture_plan, snippet_factory);

    return fcomp_handler->process(generator, pool, "lib" + sdfg.name());
}

} // namespace docc::qant
