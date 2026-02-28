#include "docc/qant/plugin.h"

#include <sdfg/analysis/analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>

sdfg::plugins::Plugin register_docc_plugin() {
    return sdfg::plugins::Plugin{
        .name = "qant",
        .version = "0.0.1",
        .description = "Q.ANT target extension for the docc compiler",
        .register_plugin_callback =
            [](sdfg::plugins::Context& context) {
                // TODO: Register Q.ANT specific library nodes
                // context.library_node_serializer_registry.register_serializer(...);

                // TODO: Register Q.ANT specific dispatchers
                // context.node_dispatcher_registry.register_dispatcher(...);

                // TODO: Register Q.ANT specific schedulers
                // context.scheduler_registry.register_scheduler(...);

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
