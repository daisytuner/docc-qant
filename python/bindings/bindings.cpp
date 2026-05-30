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
}
