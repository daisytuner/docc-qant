#include "so_compile.h"

#include <docc/compile/src_file_compiler.h>
#include <docc/compile/src_file_compiler_builder.h>
#include <docc/target/docc_target.h>
#include <gtest/gtest.h>
#include <sdfg/codegen/code_generators/cpp_code_generator.h>

#include "docc/qant/plugin.h"
#include "sdfg_debug_dump.h"

namespace docc::qant {

sdfg::plugins::Context docc_context = sdfg::plugins::Context::global_context();

static std::unique_ptr<docc::util::DefaultDoccPaths> find_docc_paths() {
    auto test = util::DefaultDoccPaths::from_lib_location(util::find_lib_location());

    return std::make_unique<
        util::DefaultDoccPaths>(DOCC_PATHS_BIN, DOCC_PATHS_SRC, util::DefaultDoccPaths::DoccRootMode::CMake);
}

static std::shared_ptr<docc::util::DefaultDoccPaths> docc_paths = find_docc_paths();

std::string so_compile(sdfg::StructuredSDFG& sdfg, sdfg::plugins::Context& context) {
    auto dir = test_output_dir ? *test_output_dir : std::filesystem::path("/tmp");
    auto info = ::testing::UnitTest::GetInstance()->current_test_info();
    auto suite_name = info->test_suite_name();
    auto test_name = info->name();
    auto base_path = dir / suite_name / test_name;
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
