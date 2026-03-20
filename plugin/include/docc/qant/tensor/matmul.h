#ifndef DOCC_QANT_TENSOR_MATMUL_H
#define DOCC_QANT_TENSOR_MATMUL_H

#include "docc/qant/dataflow/library_nodes/math/tensor/matmul_node.h"
#include "docc/qant/passes/reduce_quantization_pass.h"
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
        const sdfg::math::tensor::QantMatMulNode& node
    );

    void dispatch_code(
        sdfg::codegen::PrettyPrinter& stream,
        sdfg::codegen::PrettyPrinter& globals_stream,
        sdfg::codegen::CodeSnippetFactory& library_snippet_factory
    ) override;

private:
    const sdfg::math::tensor::QantMatMulNode& matmul_node_;

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
    std::string calculate_tensor_start_offset(
        sdfg::symbolic::Expression tensor_offset,
        const sdfg::symbolic::MultiExpression& strides,
        size_t batch_dims,
        size_t max_batch_dims,
        std::vector<std::string> batch_vars
    );

protected:
    void alloc_arr(
        sdfg::codegen::PrettyPrinter& stream,
        sdfg::codegen::LanguageExtension& lang_ext,
        sdfg::symbolic::Expression size,
        const std::string& var,
        std::vector<std::string>& tmp_allocs
    );
};

} // namespace tensor
} // namespace qant
} // namespace docc

#endif // DOCC_QANT_TENSOR_MATMUL_H
