#include "docc/qant/tensor/sigmoid_dispatcher.h"

namespace docc {
namespace qant {
namespace tensor {

using namespace sdfg;

SigmoidNodeDispatcher_QANT::SigmoidNodeDispatcher_QANT(
    sdfg::codegen::LanguageExtension& language_extension,
    const sdfg::Function& function,
    const sdfg::data_flow::DataFlowGraph& data_flow_graph,
    const sdfg::math::tensor::SigmoidNode& node
)
    : UnaryElementWiseBaseDispatcher(language_extension, function, data_flow_graph, node, node) {}

void SigmoidNodeDispatcher_QANT::emit_toolkit_call(sdfg::codegen::PrettyPrinter& stream) {
    stream << "DLManagedTensorVersioned* __qant_result = qant_native_computing_toolkit::sigmoid_fprop(" << std::endl;
    stream << "    __qant_npu_id," << std::endl;
    stream << "    &__qant_tensor_X" << std::endl;
    stream << ");" << std::endl;
}

std::string SigmoidNodeDispatcher_QANT::toolkit_function_name() const { return "sigmoid_fprop"; }

} // namespace tensor
} // namespace qant
} // namespace docc
