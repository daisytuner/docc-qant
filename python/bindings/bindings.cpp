#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <fstream>

#include <sdfg/plugins/plugins.h>
#include <sdfg/structured_sdfg.h>

#include "docc/qant/plugin.h"

namespace py = pybind11;

PYBIND11_MODULE(_qant, m) {
    static sdfg::plugins::Context docc_context = sdfg::plugins::Context::global_context();
    sdfg::codegen::register_default_dispatchers();
    sdfg::serializer::register_default_serializers();
    docc::qant::register_plugin(docc_context);

    m.doc() = "Q.ANT target extension for the docc compiler";

    m.def(
        "register_plugin_qant",
        [](uintptr_t context_ptr) {
            auto* context = reinterpret_cast<sdfg::plugins::Context*>(context_ptr);
            sdfg::plugins::Plugin plugin = register_docc_plugin();
            plugin.register_plugin_callback(*context);
        },
        py::arg("context_ptr"),
        "Register the Q.ANT plugin with the docc compiler. Takes native pointer to context."
    );

    m.def(
        "schedule_qant",
        [](uintptr_t sdfg_ptr, const std::string& category) {
            auto* sdfg = reinterpret_cast<sdfg::StructuredSDFG*>(sdfg_ptr);
            docc::qant::schedule(*sdfg, category);
        },
        py::arg("sdfg_ptr"),
        py::arg("category"),
        "Schedule an SDFG for Q.ANT target. Takes native pointer to StructuredSDFG."
    );

    m.def(
        "expand_qant",
        [](uintptr_t sdfg_ptr, const std::string& category) {
            auto* sdfg = reinterpret_cast<sdfg::StructuredSDFG*>(sdfg_ptr);
            docc::qant::expand(*sdfg);
        },
        py::arg("sdfg_ptr"),
        py::arg("category"),
        "Schedule an SDFG for Q.ANT target. Takes native pointer to StructuredSDFG."
    );

    m.def(
        "qant_compile_hook",
        [](uintptr_t sdfg_ptr,
           const std::string& output_folder,
           const std::string& target,
           const std::string& instrumentation_mode,
           bool capture_args) {
            auto* sdfg = reinterpret_cast<sdfg::StructuredSDFG*>(sdfg_ptr);
            return docc::qant::before_compile_hook(*sdfg, output_folder, target, instrumentation_mode, capture_args);
        },
        py::arg("sdfg_ptr"),
        py::arg("output_folder"),
        py::arg("target"),
        py::arg("instrumentation_mode"),
        py::arg("capture_args"),
        "Get the final SDFG directly before compiling for last-minute changes or analysis"
    );
}
