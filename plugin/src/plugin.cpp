#include "docc/qant/plugin.h"
#include "docc/qant/blas/gemm.h"
#include "docc/qant/qant.h"

#include <sdfg/analysis/analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/data_flow/library_nodes/math/blas/gemm_node.h>

sdfg::plugins::Plugin register_docc_plugin() {
    return sdfg::plugins::Plugin{
        .name = "qant",
        .version = "0.0.1",
        .description = "Q.ANT target extension for the docc compiler",
        .register_plugin_callback =
            [](sdfg::plugins::Context& context) {
                // Register Q.ANT GEMM dispatcher
                context.library_node_dispatcher_registry.register_library_node_dispatcher(
                    sdfg::math::blas::LibraryNodeType_GEMM.value() + "::" + docc::qant::ImplementationType_QANT.value(),
                    [](sdfg::codegen::LanguageExtension& language_extension,
                       const sdfg::Function& function,
                       const sdfg::data_flow::DataFlowGraph& data_flow_graph,
                       const sdfg::data_flow::LibraryNode& node) {
                        return std::make_unique<docc::qant::blas::GEMMNodeDispatcher_QANT>(
                            language_extension,
                            function,
                            data_flow_graph,
                            dynamic_cast<const sdfg::math::blas::GEMMNode&>(node)
                        );
                    }
                );

                std::cout << "Q.ANT plugin registered with docc compiler!" << std::endl;
            },
        .sdfg_lookup = nullptr
    };
}

namespace docc {
namespace qant {

void schedule(sdfg::StructuredSDFG& sdfg, const std::string& category) {
    sdfg::builder::StructuredSDFGBuilder builder(sdfg);
    sdfg::analysis::AnalysisManager analysis_manager(sdfg);

    std::cout << "Scheduling for Q.ANT target with category: " << category << std::endl;

    // TODO: Add Q.ANT specific scheduling passes here
    // Example: Process library nodes, apply Q.ANT transformations
}

} // namespace qant
} // namespace docc
