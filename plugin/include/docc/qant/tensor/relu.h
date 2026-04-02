#ifndef DOCC_QANT_TENSOR_RELU_H
#define DOCC_QANT_TENSOR_RELU_H

#include "docc/qant/dataflow/library_nodes/math/tensor/elementwise_ops/relu_node.h"
#include "docc/qant/tensor/qant_tensor_dispatcher.h"
#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/relu_node.h"

namespace docc {
namespace qant {
namespace tensor {

class ReLUNodeDispatcher_QANT : public QantTensorLibNodeDispatcher {
public:
    ReLUNodeDispatcher_QANT(
        sdfg::codegen::LanguageExtension& language_extension,
        const sdfg::Function& function,
        const sdfg::data_flow::DataFlowGraph& data_flow_graph,
        const sdfg::math::tensor::QantReLUNode& node
    );

    void dispatch_code(
        sdfg::codegen::PrettyPrinter& stream,
        sdfg::codegen::PrettyPrinter& globals_stream,
        sdfg::codegen::CodeSnippetFactory& library_snippet_factory
    ) override;

private:
    const sdfg::math::tensor::QantReLUNode& relu_node_;
};

} // namespace tensor
} // namespace qant
} // namespace docc

#endif // DOCC_QANT_TENSOR_RELU_H
