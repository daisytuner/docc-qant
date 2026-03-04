#pragma once

#include <sdfg/plugins/plugins.h>

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

} // namespace qant
} // namespace docc
