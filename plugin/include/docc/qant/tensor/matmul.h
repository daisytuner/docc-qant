#ifndef DOCC_QANT_TENSOR_MATMUL_H
#define DOCC_QANT_TENSOR_MATMUL_H

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"

namespace docc {
namespace qant {
namespace tensor {

class MatMulNodeDispatcher_QANT : public sdfg::codegen::LibraryNodeDispatcher {

public:
    MatMulNodeDispatcher_QANT(
        sdfg::codegen::LanguageExtension& language_extension,
        const sdfg::Function& function,
        const sdfg::data_flow::DataFlowGraph& data_flow_graph,
        const sdfg::math::tensor::MatMulNode& node
    );

    void dispatch_code(
        sdfg::codegen::PrettyPrinter& stream,
        sdfg::codegen::PrettyPrinter& globals_stream,
        sdfg::codegen::CodeSnippetFactory& library_snippet_factory
    ) override;

private:
    const sdfg::math::tensor::MatMulNode& matmul_node_;

    void emit_dlpack_tensor_wrapper(
        sdfg::codegen::PrettyPrinter& stream,
        const std::string& var_name,
        const std::string& data_ptr,
        const std::string& rows,
        const std::string& cols,
        const std::string& stride_row,
        const std::string& stride_col,
        const std::string& offset
    );
};

} // namespace tensor
} // namespace qant
} // namespace docc

#endif // DOCC_QANT_TENSOR_MATMUL_H
