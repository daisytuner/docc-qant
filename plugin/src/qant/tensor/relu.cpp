#include "docc/qant/tensor/relu.h"

namespace docc {
namespace qant {
namespace tensor {

using namespace sdfg;

ReLUNodeDispatcher_QANT::ReLUNodeDispatcher_QANT(
    sdfg::codegen::LanguageExtension& language_extension,
    const sdfg::Function& function,
    const sdfg::data_flow::DataFlowGraph& data_flow_graph,
    const sdfg::math::tensor::ReLUNode& node
)
    : UnaryElementWiseBaseDispatcher(language_extension, function, data_flow_graph, node, node) {}

void ReLUNodeDispatcher_QANT::emit_toolkit_call(sdfg::codegen::PrettyPrinter& stream) {
    stream << "DLManagedTensorVersioned* __qant_result = qant_native_computing_toolkit::relu_fprop(" << std::endl;
    stream << "    __qant_npu_id," << std::endl;
    stream << "    &__qant_tensor_X" << std::endl;
    stream << ");" << std::endl;
}

std::string ReLUNodeDispatcher_QANT::toolkit_function_name() const { return "relu_fprop"; }

} // namespace tensor
} // namespace qant
} // namespace docc
