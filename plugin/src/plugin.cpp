#include "docc/qant/plugin.h"
#include "docc/qant/blas/gemm.h"
#include "docc/qant/dataflow/library_nodes/math/tensor/conv_node.h"
#include "docc/qant/dataflow/library_nodes/math/tensor/elementwise_ops/qant_elementwise_base_serializer.h"
#include "docc/qant/dataflow/library_nodes/math/tensor/elementwise_ops/relu_node.h"
#include "docc/qant/dataflow/library_nodes/math/tensor/elementwise_ops/sigmoid_node.h"
#include "docc/qant/dataflow/library_nodes/math/tensor/matmul_node.h"
#include "docc/qant/dataflow/library_nodes/math/tensor/pooling_node.h"
#include "docc/qant/passes/remapping_pass.h"
#include "docc/qant/qant.h"
#include "docc/qant/qant_offloaded_visitor.h"
#include "docc/qant/tensor/conv.h"
#include "docc/qant/tensor/matmul.h"
#include "docc/qant/tensor/pooling.h"
#include "docc/qant/tensor/relu.h"
#include "sdfg/codegen/code_generators/cpp_code_generator.h"
#include "sdfg/passes/einsum.h"
#include "sdfg/passes/pipeline.h"
#include "sdfg/structured_sdfg.h"

#include <dlfcn.h>
#include <iomanip>
#include <optional>
#include <sdfg/analysis/analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/data_flow/library_nodes/math/blas/gemm_node.h>
#include <sdfg/data_flow/library_nodes/math/tensor/conv_node.h>
#include <sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/relu_node.h>
#include <sdfg/data_flow/library_nodes/math/tensor/matmul_node.h>
#include <sdfg/data_flow/library_nodes/math/tensor/pooling_node.h>
#include <sdfg/passes/targets/target_mapping_pass.h>

#include "docc/qant/tensor/batchnorm_dispatcher.h"
#include "docc/qant/tensor/sigmoid_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/tensor/batchnorm_node.h"
#include "sdfg/visualizer/dot_visualizer.h"

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
                language_extension,
                function,
                data_flow_graph,
                dynamic_cast<const sdfg::math::tensor::QantConvNode&>(node)
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

    // Register Q.ANT ReLU dispatcher
    context.library_node_dispatcher_registry.register_library_node_dispatcher(
        sdfg::math::tensor::LibraryNodeType_QantReLU.value() + "::" + docc::qant::ImplementationType_QANT.value(),
        [](sdfg::codegen::LanguageExtension& language_extension,
           const sdfg::Function& function,
           const sdfg::data_flow::DataFlowGraph& data_flow_graph,
           const sdfg::data_flow::LibraryNode& node) {
            return std::make_unique<docc::qant::tensor::ReLUNodeDispatcher_QANT>(
                language_extension,
                function,
                data_flow_graph,
                dynamic_cast<const sdfg::math::tensor::QantReLUNode&>(node)
            );
        }
    );

    // Register Q.ANT Sigmoid dispatcher
    context.library_node_dispatcher_registry.register_library_node_dispatcher(
        sdfg::math::tensor::LibraryNodeType_QantSigmoid.value() + "::" + docc::qant::ImplementationType_QANT.value(),
        [](sdfg::codegen::LanguageExtension& language_extension,
           const sdfg::Function& function,
           const sdfg::data_flow::DataFlowGraph& data_flow_graph,
           const sdfg::data_flow::LibraryNode& node) {
            return std::make_unique<docc::qant::tensor::SigmoidNodeDispatcher_QANT>(
                language_extension,
                function,
                data_flow_graph,
                dynamic_cast<const sdfg::math::tensor::QantSigmoidNode&>(node)
            );
        }
    );

    // Register Q.ANT Batchnorm dispatcher
    context.library_node_dispatcher_registry.register_library_node_dispatcher(
        sdfg::math::tensor::LibraryNodeType_BatchNorm.value() + "::" + docc::qant::ImplementationType_QANT.value(),
        [](sdfg::codegen::LanguageExtension& language_extension,
           const sdfg::Function& function,
           const sdfg::data_flow::DataFlowGraph& data_flow_graph,
           const sdfg::data_flow::LibraryNode& node) {
            return std::make_unique<docc::qant::tensor::BatchnormNodeDispatcher_QANT>(
                language_extension,
                function,
                data_flow_graph,
                dynamic_cast<const sdfg::math::tensor::BatchNormNode&>(node)
            );
        }
    );

    // Register QantReLU serializer
    context.library_node_serializer_registry
        .register_library_node_serializer(sdfg::math::tensor::LibraryNodeType_QantReLU.value(), []() {
            return std::make_unique<
                sdfg::math::tensor::QantElementWiseBaseSerializer<sdfg::math::tensor::QantReLUNode>>();
        });

    // Register QantSigmoid serializer
    context.library_node_serializer_registry
        .register_library_node_serializer(sdfg::math::tensor::LibraryNodeType_QantSigmoid.value(), []() {
            return std::make_unique<
                sdfg::math::tensor::QantElementWiseBaseSerializer<sdfg::math::tensor::QantSigmoidNode>>();
        });

    std::cout << "Q.ANT plugin registered with docc compiler!" << std::endl;
};

void expand(sdfg::StructuredSDFG& sdfg) {
    sdfg::builder::StructuredSDFGBuilder builder(sdfg);
    sdfg::analysis::AnalysisManager analysis_manager(sdfg);

    // Run expansion pass
    sdfg::passes::QantRemappingPass remapping;
    remapping.run(builder, analysis_manager);

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

/**
 * @brief Try to compute a concrete percentage from two symbolic expressions.
 *
 * Returns a numeric percentage (part / whole * 100) only when both expressions
 * evaluate to plain integers.  If either is null, zero-denominator, or contains
 * free symbols the result is std::nullopt — a symbolic ratio would not be
 * human-readable anyway.
 */
static std::optional<double>
try_compute_percentage(const sdfg::symbolic::Expression& part, const sdfg::symbolic::Expression& whole) {
    if (part.is_null() || whole.is_null()) return std::nullopt;
    if (!SymEngine::is_a<SymEngine::Integer>(*part)) return std::nullopt;
    if (!SymEngine::is_a<SymEngine::Integer>(*whole)) return std::nullopt;

    auto whole_int = SymEngine::down_cast<const SymEngine::Integer&>(*whole).as_int();
    if (whole_int == 0) return std::nullopt;

    auto part_int = SymEngine::down_cast<const SymEngine::Integer&>(*part).as_int();
    return static_cast<double>(part_int) / static_cast<double>(whole_int) * 100.0;
}

void analyze_offloading(sdfg::StructuredSDFG& sdfg) {
    sdfg::analysis::AnalysisManager analysis_manager(sdfg);
    auto& flop_ana = analysis_manager.get<sdfg::analysis::FlopAnalysis>();

    // ── Total FLOPs (from whole-program analysis) ───────────────────────
    auto total = flop_ana.get(&sdfg.root());
    std::cout << "=== Offloading Analysis ===" << std::endl;
    std::cout << "  Total FLOPs: ";
    if (!total.is_null()) {
        std::cout << total->__str__();
    } else {
        std::cout << "N/A (could not be determined)";
    }
    std::cout << std::endl;

    // ── Collect Qant-offloaded nodes ────────────────────────────────────
    QantOffloadedVisitor visitor;
    visitor.dispatch(sdfg.root());

    const auto& offloaded = visitor.offloaded();
    if (offloaded.empty()) {
        std::cout << "  No Qant-offloaded library nodes found." << std::endl;
        std::cout << "===========================" << std::endl;
        return;
    }

    std::cout << "  Offloaded library nodes: " << offloaded.size() << std::endl;

    // ── Accumulate offloaded FLOPs, grouped by operator code ────────────
    // Preserves insertion order via a vector of keys.
    struct GroupAccum {
        sdfg::symbolic::Expression flops = sdfg::symbolic::zero();
        size_t count = 0;
        bool available = true;
    };
    std::vector<std::string> group_order;
    std::unordered_map<std::string, GroupAccum> groups;

    sdfg::symbolic::Expression offloaded_flops = sdfg::symbolic::zero();
    bool offloaded_available = true;

    for (const auto& info : offloaded) {
        std::string node_name = info.node->toStr();
        std::string op_code = info.node->code().value();

        // Flag loop nesting — we don't expect this yet and would need to
        // account for trip counts if it ever happens.
        if (info.enclosing_loop != nullptr) {
            std::cout << "    [WARN] " << node_name << "  — inside a loop; offloaded flop total may be inaccurate"
                      << std::endl;
        }

        // Ensure the group exists and track insertion order.
        if (groups.find(op_code) == groups.end()) {
            group_order.push_back(op_code);
        }
        auto& group = groups[op_code];
        group.count++;

        auto node_flops = info.node->flop();
        if (node_flops.is_null()) {
            std::cout << "    [!] " << node_name << "  — flop count unavailable" << std::endl;
            offloaded_available = false;
            group.available = false;
            continue;
        }

        offloaded_flops = sdfg::symbolic::add(offloaded_flops, node_flops);
        group.flops = sdfg::symbolic::add(group.flops, node_flops);
    }

    // ── Per-operator subtotals ──────────────────────────────────────────
    std::cout << "  Per-operator breakdown:" << std::endl;
    for (const auto& op_code : group_order) {
        const auto& group = groups.at(op_code);
        std::cout << "    " << op_code << " (" << group.count << " node" << (group.count != 1 ? "s" : "") << "): ";
        if (group.available) {
            std::cout << group.flops->__str__();
        } else {
            std::cout << "incomplete (" << group.flops->__str__() << " + unknown)";
        }
        if (group.available) {
            auto pct = try_compute_percentage(group.flops, total);
            if (pct.has_value()) {
                std::cout << std::fixed << std::setprecision(4) << "  (" << pct.value() << "% of total)";
            }
        }
        std::cout << std::endl;
    }

    // ── Summary ─────────────────────────────────────────────────────────
    std::cout << "  Offloaded FLOPs: ";
    if (offloaded_available) {
        std::cout << offloaded_flops->__str__();
    } else {
        std::cout << "incomplete (" << offloaded_flops->__str__() << " + unknown contributions)";
    }
    std::cout << std::endl;

    std::cout << "  Offloaded ratio: ";
    if (offloaded_available) {
        auto pct = try_compute_percentage(offloaded_flops, total);
        if (pct.has_value()) {
            std::cout << std::fixed << std::setprecision(4) << pct.value() << " %" << std::endl;
        } else {
            std::cout << "N/A (symbolic — cannot compute numeric ratio)" << std::endl;
        }
    } else {
        std::cout << "N/A (missing flop information)" << std::endl;
    }

    std::cout << "===========================" << std::endl;
}

std::string compile(
    sdfg::StructuredSDFG& sdfg,
    const std::string& output_folder,
    const std::string& target,
    const std::string& instrumentation_mode,
    bool capture_args
) {
    analyze_offloading(sdfg);
    // All we need is: use g++, add -std=c++23, and link against the -lqant_native_computing_toolkit in terms of changes
    // from the base docc compilation flow. It is scheduled for that code to become more modular, such that it can be
    // called from here, just with additional options to override the options we need

    fs::path build_path(output_folder);
    if (!fs::exists(build_path)) {
        fs::create_directories(build_path);
    }
    fs::path header_path = build_path / (sdfg.name() + ".h");
    fs::path source_path = build_path / (sdfg.name() + ".cpp");

    sdfg::analysis::AnalysisManager analysis_manager(sdfg);

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
