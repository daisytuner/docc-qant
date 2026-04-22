#pragma once
#include "sdfg/plugins/plugins.h"

void dump_sdfg(const sdfg::StructuredSDFG& sdfg, const std::string& step);

namespace docc::qant {
extern sdfg::plugins::Context& docc_context;

std::string so_compile(sdfg::StructuredSDFG& sdfg, sdfg::plugins::Context& context = docc_context);
} // namespace docc::qant
