#include "docc/qant/plugin.h"
#include "docc/qant/blas/gemm.h"
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
#include <sdfg/plugins/targets.h>

#include "docc/qant/tensor/batchnorm_dispatcher.h"
#include "docc/qant/tensor/sigmoid_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/tensor/batchnorm_node.h"
#include "sdfg/visualizer/dot_visualizer.h"

#include <docc/compile/src_file_compiler_builder.h>
#include <docc/target/docc_target.h>

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

docc::target::DoccTarget qant_target{
    .api_ver = docc::target::DoccTarget::NEWEST_API_VER,
    .short_name = "qant",
    .apply_additional_compile_options = [](compile::SrcFileCompilerBuilder& builder) -> bool {
        builder.set_compiler("g++");
        builder.add_compile_option("-std=c++23");
        builder.add_link_option("-lqant_native_computing_toolkit");
        return true;
    },
    .apply_expand_time_mapping = [](sdfg::builder::StructuredSDFGBuilder& builder,
                                    sdfg::analysis::AnalysisManager& analysis_manager,
                                    const docc::target::TargetOptions& options) -> bool {
        // Run expansion pass
        sdfg::passes::QantRemappingPass remapping;
        remapping.run(builder, analysis_manager);

        ReduceQuantizationPass reduceQuantizationPass;
        reduceQuantizationPass.run_pass(builder, analysis_manager);
        return true;
    },
};

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
        sdfg::math::tensor::LibraryNodeType_MatMul.value() + "::" + docc::qant::ImplementationType_QANT.value(),
        [](sdfg::codegen::LanguageExtension& language_extension,
           const sdfg::Function& function,
           const sdfg::data_flow::DataFlowGraph& data_flow_graph,
           const sdfg::data_flow::LibraryNode& node) {
            return std::make_unique<docc::qant::tensor::MatMulNodeDispatcher_QANT>(
                language_extension,
                function,
                data_flow_graph,
                dynamic_cast<const sdfg::math::tensor::MatMulNode&>(node)
            );
        }
    );



    // Register Q.ANT Conv dispatcher
    context.library_node_dispatcher_registry.register_library_node_dispatcher(
        sdfg::math::tensor::LibraryNodeType_Conv.value() + "::" + docc::qant::ImplementationType_QANT.value(),
        [](sdfg::codegen::LanguageExtension& language_extension,
           const sdfg::Function& function,
           const sdfg::data_flow::DataFlowGraph& data_flow_graph,
           const sdfg::data_flow::LibraryNode& node) {
            return std::make_unique<docc::qant::tensor::ConvNodeDispatcher_QANT>(
                language_extension,
                function,
                data_flow_graph,
                dynamic_cast<const sdfg::math::tensor::ConvNode&>(node)
            );
        }
    );

    // Register Q.ANT Pooling dispatcher
    context.library_node_dispatcher_registry.register_library_node_dispatcher(
        sdfg::math::tensor::LibraryNodeType_Pooling.value() + "::" + docc::qant::ImplementationType_QANT.value(),
        [](sdfg::codegen::LanguageExtension& language_extension,
           const sdfg::Function& function,
           const sdfg::data_flow::DataFlowGraph& data_flow_graph,
           const sdfg::data_flow::LibraryNode& node) {
            return std::make_unique<docc::qant::tensor::PoolingNodeDispatcher_QANT>(
                language_extension, function, data_flow_graph, dynamic_cast<const sdfg::math::tensor::PoolingNode&>(node)
            );
        }
    );

    // Register Q.ANT ReLU dispatcher
    context.library_node_dispatcher_registry.register_library_node_dispatcher(
        sdfg::math::tensor::LibraryNodeType_ReLU.value() + "::" + docc::qant::ImplementationType_QANT.value(),
        [](sdfg::codegen::LanguageExtension& language_extension,
           const sdfg::Function& function,
           const sdfg::data_flow::DataFlowGraph& data_flow_graph,
           const sdfg::data_flow::LibraryNode& node) {
            return std::make_unique<docc::qant::tensor::ReLUNodeDispatcher_QANT>(
                language_extension,
                function,
                data_flow_graph,
                dynamic_cast<const sdfg::math::tensor::ReLUNode&>(node)
            );
        }
    );

    // Register Q.ANT Sigmoid dispatcher
    context.library_node_dispatcher_registry.register_library_node_dispatcher(
        sdfg::math::tensor::LibraryNodeType_Sigmoid.value() + "::" + docc::qant::ImplementationType_QANT.value(),
        [](sdfg::codegen::LanguageExtension& language_extension,
           const sdfg::Function& function,
           const sdfg::data_flow::DataFlowGraph& data_flow_graph,
           const sdfg::data_flow::LibraryNode& node) {
            return std::make_unique<docc::qant::tensor::SigmoidNodeDispatcher_QANT>(
                language_extension,
                function,
                data_flow_graph,
                dynamic_cast<const sdfg::math::tensor::SigmoidNode&>(node)
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


    context.add_target(&qant_target);
};

void expand(sdfg::StructuredSDFG& sdfg) {}

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

void before_compile_hook(
    sdfg::StructuredSDFG& sdfg,
    const std::string& output_folder,
    const std::string& target,
    const std::string& instrumentation_mode,
    bool capture_args
) {
    analyze_offloading(sdfg);
}


} // namespace qant
} // namespace docc
