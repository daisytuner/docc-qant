#pragma once

#include "docc/qant/tensor/unary_elementwise_base_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/sigmoid_node.h"

namespace docc {
namespace qant {
namespace tensor {

class SigmoidNodeDispatcher_QANT : public UnaryElementWiseBaseDispatcher {
public:
    SigmoidNodeDispatcher_QANT(
        sdfg::codegen::LanguageExtension& language_extension,
        const sdfg::Function& function,
        const sdfg::data_flow::DataFlowGraph& data_flow_graph,
        const sdfg::math::tensor::SigmoidNode& node
    );

protected:
    void emit_toolkit_call(sdfg::codegen::PrettyPrinter& stream) override;
    std::string toolkit_function_name() const override;
};

} // namespace tensor
} // namespace qant
} // namespace docc
