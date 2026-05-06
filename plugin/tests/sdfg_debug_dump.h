#pragma once

#include <filesystem>
#include <sdfg/structured_sdfg.h>

extern std::optional<std::filesystem::path> test_output_dir;

void dump_sdfg(const sdfg::StructuredSDFG& sdfg, const std::string& step);
