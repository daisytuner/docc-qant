#pragma once

#include <sdfg/plugins/plugins.h>
#include <sdfg/plugins/target_mapping.h>

// Plugin registration at top-level namespace
sdfg::plugins::Plugin register_docc_plugin();

namespace docc {
namespace qant {

void register_plugin(sdfg::plugins::Context& context);

void schedule(sdfg::StructuredSDFG& sdfg, const std::string& category);

std::string compile(
    sdfg::StructuredSDFG& sdfg,
    const std::string& output_folder,
    const std::string& target,
    const std::string& instrumentation_mode,
    bool capture_args
);

void expand(sdfg::StructuredSDFG& sdfg);


class QantLibNodeMapper : public sdfg::plugins::TargetMapper {
public:
    bool try_map(
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::analysis::AnalysisManager& analysis_manager,
        sdfg::data_flow::LibraryNode& node
    ) const override;
};

} // namespace qant
} // namespace docc
