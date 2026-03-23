#include "docc/qant/plugin.h"
#include "docc/qant/blas/gemm.h"
#include "docc/qant/dataflow/library_nodes/math/tensor/conv_node.h"
#include "docc/qant/dataflow/library_nodes/math/tensor/matmul_node.h"
#include "docc/qant/dataflow/library_nodes/math/tensor/pooling_node.h"
#include "docc/qant/passes/remapping_pass.h"
#include "docc/qant/qant.h"
#include "docc/qant/tensor/conv.h"
#include "docc/qant/tensor/matmul.h"
#include "docc/qant/tensor/pooling.h"
#include "sdfg/codegen/code_generators/cpp_code_generator.h"
#include "sdfg/passes/pipeline.h"
#include "sdfg/structured_sdfg.h"

#include <dlfcn.h>
#include <sdfg/analysis/analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/data_flow/library_nodes/math/blas/gemm_node.h>
#include <sdfg/data_flow/library_nodes/math/tensor/conv_node.h>
#include <sdfg/data_flow/library_nodes/math/tensor/matmul_node.h>
#include <sdfg/data_flow/library_nodes/math/tensor/pooling_node.h>
#include <sdfg/passes/targets/target_mapping_pass.h>

sdfg::plugins::Plugin register_docc_plugin() {
    return sdfg::plugins::Plugin{
        .name = "qant",
        .version = "0.0.1",
        .description = "Q.ANT target extension for the docc compiler",
        .register_plugin_callback = docc::qant::register_plugin,
        .sdfg_lookup = nullptr
    };
}

namespace docc {
namespace qant {

void register_plugin(sdfg::plugins::Context& context) {
    // Register Q.ANT GEMM dispatcher
    context.library_node_dispatcher_registry.register_library_node_dispatcher(
        sdfg::math::blas::LibraryNodeType_GEMM.value() + "::" + docc::qant::ImplementationType_QANT.value(),
        [](sdfg::codegen::LanguageExtension& language_extension,
           const sdfg::Function& function,
           const sdfg::data_flow::DataFlowGraph& data_flow_graph,
           const sdfg::data_flow::LibraryNode& node) {
            return std::make_unique<docc::qant::blas::GEMMNodeDispatcher_QANT>(
                language_extension, function, data_flow_graph, dynamic_cast<const sdfg::math::blas::GEMMNode&>(node)
            );
        }
    );

    // Register Q.ANT MatMul dispatcher
    context.library_node_dispatcher_registry.register_library_node_dispatcher(
        sdfg::math::tensor::LibraryNodeType_QantMatMul.value() + "::" + docc::qant::ImplementationType_QANT.value(),
        [](sdfg::codegen::LanguageExtension& language_extension,
           const sdfg::Function& function,
           const sdfg::data_flow::DataFlowGraph& data_flow_graph,
           const sdfg::data_flow::LibraryNode& node) {
            return std::make_unique<docc::qant::tensor::MatMulNodeDispatcher_QANT>(
                language_extension,
                function,
                data_flow_graph,
                dynamic_cast<const sdfg::math::tensor::QantMatMulNode&>(node)
            );
        }
    );

    // Register QantMatMul serializer
    context.library_node_serializer_registry
        .register_library_node_serializer(sdfg::math::tensor::LibraryNodeType_QantMatMul.value(), []() {
            return std::make_unique<sdfg::math::tensor::QantMatMulNodeSerializer>();
        });

    // Register Q.ANT Conv dispatcher
    context.library_node_dispatcher_registry.register_library_node_dispatcher(
        sdfg::math::tensor::LibraryNodeType_QantConv.value() + "::" + docc::qant::ImplementationType_QANT.value(),
        [](sdfg::codegen::LanguageExtension& language_extension,
           const sdfg::Function& function,
           const sdfg::data_flow::DataFlowGraph& data_flow_graph,
           const sdfg::data_flow::LibraryNode& node) {
            return std::make_unique<docc::qant::tensor::ConvNodeDispatcher_QANT>(
                language_extension, function, data_flow_graph, dynamic_cast<const sdfg::math::tensor::ConvNode&>(node)
            );
        }
    );

    // Register QantConv serializer
    context.library_node_serializer_registry
        .register_library_node_serializer(sdfg::math::tensor::LibraryNodeType_QantConv.value(), []() {
            return std::make_unique<sdfg::math::tensor::QantConvNodeSerializer>();
        });

    // Register Q.ANT Pooling dispatcher
    context.library_node_dispatcher_registry.register_library_node_dispatcher(
        sdfg::math::tensor::LibraryNodeType_QantPooling.value() + "::" + docc::qant::ImplementationType_QANT.value(),
        [](sdfg::codegen::LanguageExtension& language_extension,
           const sdfg::Function& function,
           const sdfg::data_flow::DataFlowGraph& data_flow_graph,
           const sdfg::data_flow::LibraryNode& node) {
            return std::make_unique<docc::qant::tensor::PoolingNodeDispatcher_QANT>(
                language_extension, function, data_flow_graph, dynamic_cast<const sdfg::math::tensor::PoolingNode&>(node)
            );
        }
    );

    // Register QantPooling serializer
    context.library_node_serializer_registry
        .register_library_node_serializer(sdfg::math::tensor::LibraryNodeType_QantPooling.value(), []() {
            return std::make_unique<sdfg::math::tensor::QantPoolingNodeSerializer>();
        });

    std::cout << "Q.ANT plugin registered with docc compiler!" << std::endl;
};

void expand(sdfg::StructuredSDFG& sdfg) {
    sdfg::builder::StructuredSDFGBuilder builder(sdfg);
    sdfg::analysis::AnalysisManager analysis_manager(sdfg);

    // Run expansion pass
    sdfg::passes::Pipeline expansion("QantRemapping");
    expansion.register_pass<sdfg::passes::QantRemappingPass>();

    expansion.run(builder, analysis_manager);

    ReduceQuantizationPass reduceQuantizationPass;
    reduceQuantizationPass.run_pass(builder, analysis_manager);
}

void schedule(sdfg::StructuredSDFG& sdfg, const std::string& category) {
    sdfg::builder::StructuredSDFGBuilder builder(sdfg);
    sdfg::analysis::AnalysisManager analysis_manager(sdfg);
}

namespace fs = std::filesystem;

#define DOCC_CXX_COMPILER "g++"

namespace {
void _anchor() {}
} // namespace

struct SnippetMetadata {
    std::string name;
    std::string extension;
};

std::string compile(
    sdfg::StructuredSDFG& sdfg,
    const std::string& output_folder,
    const std::string& target,
    const std::string& instrumentation_mode,
    bool capture_args
) {
    // All we need is: use g++, add -std=c++23, and link against the -lqant_native_computing_toolkit in terms of changes
    // from the base docc compilation flow. It is scheduled for that code to become more modular, such that it can be
    // called from here, just with additional options to override the options we need

    auto opts = std::getenv("DOCC_DEBUG");
    bool debug_build = opts != nullptr && std::string(opts).find("-g") != std::string::npos;

    fs::path build_path(output_folder);
    if (!fs::exists(build_path)) {
        fs::create_directories(build_path);
    }
    fs::path header_path = build_path / (sdfg.name() + ".h");
    fs::path source_path = build_path / (sdfg.name() + ".cpp");

    sdfg::analysis::AnalysisManager analysis_manager(sdfg);

    // Run expansion pass
    sdfg::passes::Pipeline expansion = sdfg::passes::Pipeline::expansion();
    sdfg::builder::StructuredSDFGBuilder builder_opt(sdfg);
    expansion.run(builder_opt, analysis_manager);

    // Instrumentation plan
    std::unique_ptr<sdfg::codegen::InstrumentationPlan> instrumentation_plan;
    if (instrumentation_mode.empty()) {
        instrumentation_plan = sdfg::codegen::InstrumentationPlan::none(sdfg);
    } else if (instrumentation_mode == "ols") {
        instrumentation_plan = sdfg::codegen::InstrumentationPlan::outermost_loops_plan(sdfg);
    } else {
        throw std::runtime_error("Unsupported instrumentation plan: " + instrumentation_mode);
    }

    // Argument capture plan
    std::unique_ptr<sdfg::codegen::ArgCapturePlan> arg_capture_plan;
    if (capture_args) {
        arg_capture_plan = sdfg::codegen::ArgCapturePlan::outermost_loops_plan(sdfg);
    } else {
        arg_capture_plan = sdfg::codegen::ArgCapturePlan::none(sdfg);
    }

    std::pair<std::filesystem::path, std::filesystem::path> lib_config = std::make_pair(build_path, header_path);
    std::shared_ptr<sdfg::codegen::CodeSnippetFactory> snippet_factory =
        std::make_shared<sdfg::codegen::CodeSnippetFactory>(&lib_config);
    sdfg::codegen::CPPCodeGenerator
        generator(sdfg, analysis_manager, *instrumentation_plan, *arg_capture_plan, snippet_factory);
    generator.generate();

    generator.as_source(header_path, source_path);

    // Write library snippets
    std::unordered_map<std::string, SnippetMetadata> lib_files;
    for (auto& [name, snippet] : snippet_factory->snippets()) {
        if (snippet.is_as_file()) {
            auto p = build_path / (name + "." + snippet.extension());
            std::ofstream outfile_lib;
            if (!lib_files.contains(p.string())) {
                outfile_lib.open(p, std::ios_base::out);
                lib_files[p.string()] = {name, snippet.extension()};
            } else {
                outfile_lib.open(p, std::ios_base::app);
            }
            if (!outfile_lib.is_open()) {
                throw std::runtime_error("Failed to open library file: " + p.string());
            }
            outfile_lib << snippet.stream().str() << std::endl;
            outfile_lib.close();
        }
    }

    // Find libraries relative to the module location
    Dl_info info;
    fs::path package_path;
    std::string package_path_str;
    std::string package_lib_path_str;
    std::string package_include_path_str;
    if (dladdr((void*) &_anchor, &info)) {
        fs::path lib_path = fs::canonical(info.dli_fname);
        package_path = lib_path.parent_path().parent_path();
        package_path_str = package_path.string();
        package_lib_path_str = (package_path / "lib").string();
        package_include_path_str = (package_path / "include").string();
    }

    // Compile
    std::unordered_set<std::string> object_files;
    for (const auto& [lib_file, meta] : lib_files) {
        std::filesystem::path lib_path(lib_file);
        auto& [snippet_name, extension] = meta;
        if (extension == "json") {
            continue;
        }

        std::string name = lib_path.stem().string();
        std::string object_file = build_path.string() + "/" + name + ".o";
        std::stringstream cmd;
        cmd << DOCC_CXX_COMPILER << " -c -fPIC -O3 -std=c++23  -march=native -mtune=native -funroll-loops";
        if (!package_path_str.empty()) {
            cmd << " -L" << package_lib_path_str;
            cmd << " -I" << package_include_path_str;
        }
#if defined(__APPLE__)
        cmd << " -I/opt/homebrew/include";
#endif

        cmd << " " << lib_file;
        if (debug_build) {
            cmd << " -g";
        }
        cmd << " -o " << object_file;
        cmd << " -lm";
        int ret = std::system(cmd.str().c_str());
        if (ret != 0) {
            throw std::runtime_error("Compilation failed: " + cmd.str());
        }
        object_files.insert(object_file);
    }

    {
        std::stringstream cmd;
        cmd << DOCC_CXX_COMPILER << " -c -fPIC -O3 -std=c++23  -march=native -mtune=native -funroll-loops";
        if (!package_path_str.empty()) {
            cmd << " -L" << package_lib_path_str;
            cmd << " -I" << package_include_path_str;
        }
        cmd << " " << source_path.string();
        if (debug_build) {
            cmd << " -g";
        }
        cmd << " -o " << (build_path / (sdfg.name() + ".o")).string();
        DEBUG_PRINTLN("Compile: " << cmd.str());
        int ret = std::system(cmd.str().c_str());
        if (ret != 0) {
            throw std::runtime_error("Compilation failed: " + cmd.str());
        }
        object_files.insert((build_path / (sdfg.name() + ".o")).string());
    }

    // Link into shared library
    fs::path lib_path = build_path / ("lib" + sdfg.name() + ".so");

    std::stringstream cmd;
#if defined(__APPLE__)
    cmd << DOCC_CXX_COMPILER << " -shared -Xpreprocessor -fopenmp -fPIC -O3";
    cmd << " -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include";
    cmd << " -L/opt/homebrew/lib";
#else
    cmd << DOCC_CXX_COMPILER << " -shared -fopenmp -fPIC -O3";
#endif
    if (!package_path_str.empty()) {
        cmd << " -L" << package_lib_path_str;
        cmd << " -I" << package_include_path_str;
    }
    // cmd << " " << source_path.string();
    for (const auto& object_file : object_files) {
        cmd << " " << object_file;
    }
    cmd << " -ldaisy_rtl";
    cmd << " -larg_capture_io";
#if defined(__APPLE__)
    cmd << " -lomp";
    cmd << " -framework Accelerate";
#else
    cmd << " -lblas";
#endif
    cmd << " -lm";
    cmd << " -lstdc++";
    if (target == "onnx") {
        cmd << " -L/usr/local/onnxruntime/lib";
        cmd << " -lonnxruntime";
        cmd << " -ldl"; // Required for dladdr()
    }
    cmd << " -lqant_native_computing_toolkit";
    cmd << " -o " << lib_path.string();

    DEBUG_PRINTLN("Link: " << cmd.str());
    int ret = std::system(cmd.str().c_str());
    if (ret != 0) {
        throw std::runtime_error("Compilation failed: " + cmd.str());
    }

    return lib_path.string();
}


} // namespace qant
} // namespace docc
