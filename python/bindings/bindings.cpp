#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <fstream>

#include <sdfg/structured_sdfg.h>
#include <sdfg/plugins/plugins.h>

#include "docc/qant/plugin.h"

namespace py = pybind11;

PYBIND11_MODULE(_qant, m) {
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
}
