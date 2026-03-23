#pragma once

#include "docc/qant/dataflow/library_nodes/math/tensor/tensor_layout.h"
#include "docc/qant/passes/reduce_quantization_pass.h"
#include "sdfg/codegen/dispatchers/block_dispatcher.h"

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
        const sdfg::math::tensor::TensorLayout& layout
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

    sdfg::math::tensor::TensorLayout transposed_layout_linear(const sdfg::math::tensor::TensorLayout& layout);

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
};

} // namespace docc::qant::tensor
