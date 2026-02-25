#ifndef DOCC_QANT_TENSOR_CONV_H
#define DOCC_QANT_TENSOR_CONV_H

#include "docc/qant/dataflow/library_nodes/math/tensor/conv_node.h"
#include "qant_tensor_dispatcher.h"
#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/tensor/conv_node.h"

namespace docc {
namespace qant {
namespace tensor {

class ConvNodeDispatcher_QANT : public QantTensorLibNodeDispatcher {
public:
    ConvNodeDispatcher_QANT(
        sdfg::codegen::LanguageExtension& language_extension,
        const sdfg::Function& function,
        const sdfg::data_flow::DataFlowGraph& data_flow_graph,
        const sdfg::math::tensor::QantConvNode& node
    );

    void dispatch_code(
        sdfg::codegen::PrettyPrinter& stream,
        sdfg::codegen::PrettyPrinter& globals_stream,
        sdfg::codegen::CodeSnippetFactory& library_snippet_factory
    ) override;

private:
    const sdfg::math::tensor::ConvNode& conv_node_;

    void emit_dlpack_tensor_wrapper_4d(
        sdfg::codegen::PrettyPrinter& stream,
        const std::string& var_name,
        const std::string& data_ptr,
        const std::string& dim0,
        const std::string& dim1,
        const std::string& dim2,
        const std::string& dim3
    );
};

} // namespace tensor
} // namespace qant
} // namespace docc

#endif // DOCC_QANT_TENSOR_CONV_H
