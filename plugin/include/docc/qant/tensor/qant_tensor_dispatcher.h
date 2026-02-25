#pragma once

#include "docc/qant/passes/reduce_quantization_pass.h"
#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"

namespace docc::qant::tensor {

class QantTensorLibNodeDispatcher : public sdfg::codegen::LibraryNodeDispatcher {
protected:
    QantTensorLibNodeDispatcher(
        sdfg::codegen::LanguageExtension& language_extension,
        const sdfg::Function& function,
        const sdfg::data_flow::DataFlowGraph& data_flow_graph,
        const sdfg::data_flow::LibraryNode& node
    );

    void emit_dlpack_tensor_wrapper(
        sdfg::codegen::PrettyPrinter& stream,
        const std::string& var_name,
        const std::string& data_ptr,
        const sdfg::math::tensor::TensorLayout& layout,
        int visible_dims
    );

    std::string calculate_tensor_start_offset(
        sdfg::symbolic::Expression tensor_offset,
        const sdfg::symbolic::MultiExpression& strides,
        size_t batch_dims,
        size_t max_batch_dims,
        std::vector<std::string> batch_vars
    );

    void alloc_arr(
        sdfg::codegen::PrettyPrinter& stream,
        sdfg::codegen::LanguageExtension& lang_ext,
        sdfg::symbolic::Expression size,
        const std::string& var,
        std::vector<std::string>& tmp_allocs
    );

    /**
     * For more than 2D: only the innermost dims are considered
     */
    sdfg::math::tensor::TensorLayout transposed_layout_linear(const sdfg::math::tensor::TensorLayout& layout);

    /**
     * Returns a layout with the same shape (last 2 dims) but row-major strides
     */
    sdfg::math::tensor::TensorLayout linear_layout_same_shape(const sdfg::math::tensor::TensorLayout& layout);

    sdfg::math::tensor::TensorLayout ensure_input_in_required_qant_format(
        CodegenOutput& output,
        sdfg::codegen::LanguageExtension& lang_ext,
        const std::string& src_var,
        const sdfg::types::IType& input_type,
        const std::string& target_var,
        const sdfg::math::tensor::TensorLayout& layout,
        sdfg::symbolic::Expression target_size,
        sdfg::types::PrimitiveType target_type,
        bool require_transposed,
        std::vector<std::string>& tmp_allocs
    );

    void emit_copy_result_back_and_cleanup(
        CodegenOutput& output,
        const sdfg::data_flow::Memlet* output_memlet,
        const std::string& result_var,
        const std::string& output_data_var,
        const std::string& out_size,
        sdfg::types::PrimitiveType required_math_type
    );

    void emit_qant_npu_id(CodegenOutput& output);

public:
    static void emit_qant_includes_once(
        sdfg::codegen::PrettyPrinter& stream, sdfg::codegen::CodeSnippetFactory& code_snippet_factory
    );

    static const sdfg::data_flow::Memlet* require_unique_output_edge(
        const sdfg::data_flow::DataFlowGraph& dflow, const sdfg::data_flow::LibraryNode& node, const std::string& conn
    );
};

} // namespace docc::qant::tensor
