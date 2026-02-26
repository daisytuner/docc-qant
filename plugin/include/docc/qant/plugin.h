#pragma once

#include <sdfg/plugins/plugins.h>

// Plugin registration at top-level namespace
sdfg::plugins::Plugin register_docc_plugin();

namespace docc {
namespace qant {

void schedule(sdfg::StructuredSDFG& sdfg, const std::string& category);

}
} // namespace docc
