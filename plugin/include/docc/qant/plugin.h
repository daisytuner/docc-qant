#pragma once

#include <sdfg/plugins/plugins.h>
#include <sdfg/plugins/target_mapping.h>

// Plugin registration at top-level namespace
sdfg::plugins::Plugin register_docc_plugin();

namespace docc {
namespace qant {

void register_plugin(sdfg::plugins::Context& context);

extern docc::target::DoccTarget qant_target;

void schedule(sdfg::StructuredSDFG& sdfg, const std::string& category);

void before_compile_hook(
    sdfg::StructuredSDFG& sdfg,
    const std::string& output_folder,
    const std::string& target,
    const std::string& instrumentation_mode,
    bool capture_args
);

void expand(sdfg::StructuredSDFG& sdfg);


} // namespace qant
} // namespace docc
