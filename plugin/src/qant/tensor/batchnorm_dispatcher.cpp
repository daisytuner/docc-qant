#include "docc/qant/tensor/batchnorm_dispatcher.h"

#include "sdfg/types/type.h"

namespace docc::qant::tensor {

using namespace sdfg;

BatchnormNodeDispatcher_QANT::BatchnormNodeDispatcher_QANT(
    sdfg::codegen::LanguageExtension& language_extension,
    const sdfg::Function& function,
    const sdfg::data_flow::DataFlowGraph& data_flow_graph,
    const sdfg::math::tensor::BatchNormNode& node
)
    : QantTensorLibNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void BatchnormNodeDispatcher_QANT::dispatch_code(
    sdfg::codegen::PrettyPrinter& stream,
    sdfg::codegen::PrettyPrinter& globals_stream,
    sdfg::codegen::CodeSnippetFactory& library_snippet_factory
) {
    QantTensorLibNodeDispatcher::emit_qant_includes_once(globals_stream, library_snippet_factory);

    auto& node = dynamic_cast<const sdfg::math::tensor::BatchNormNode&>(node_);
    auto& dflow = node_.get_parent();

    //{"Batch", "Var", "E", "Gamma", "Beta", "epsilon", "B_out"},
    auto* batch_in_memlet = dflow.in_edge_for_connector(node_, "Batch");
    // auto* means_in_memlet = dflow.in_edge_for_connector(node_, "E");
    // auto* variances_in_memlet = dflow.in_edge_for_connector(node_, "Var");
    // auto* batch_in_memlet = dflow.in_edge_for_connector(node_, "Batch");
    auto* output_memlet = dflow.in_edge_for_connector(node_, "B_out");

    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    CodegenOutput output{.main = stream, .globals = globals_stream, .library_snippet_factory = library_snippet_factory};

    emit_qant_npu_id(output);

    auto& batch_layout = node.batch_layout();
    auto out_size = SymEngine::mul(batch_layout.shape());
    auto out_size_str = language_extension_.expression(out_size);

    std::vector<std::string> temp_allocs;

    auto batched_data = "__qant_batched_data";
    auto inner_size = sdfg::symbolic::mul(batch_layout.get_dim_innermost(0), batch_layout.get_dim_innermost(1));

    sdfg::math::tensor::TensorLayout featureLayout({node.num_features()}, {sdfg::symbolic::one()});


    emit_dlpack_tensor_wrapper(stream, "__qant_means", "E", featureLayout, 1);
    emit_dlpack_tensor_wrapper(stream, "__qant_variance", "Var", featureLayout, 1);
    emit_dlpack_tensor_wrapper(stream, "__qant_weights", "Gamma", featureLayout, 1);
    emit_dlpack_tensor_wrapper(stream, "__qant_bias", "Beta", featureLayout, 1);

    auto result_var = "__qant_result";
    auto batch_stride = batch_layout.strides().at(0);
    auto batch_stride_str = language_extension_.expression(batch_stride);

    auto per_batch_src = "current_batch_input";
    auto per_batch_buf = "current_batch_buf";
    stream << language_extension_.declaration(per_batch_src, batch_in_memlet->base_type()) << " = Batch;" << std::endl;
    stream << "for (int b=0; b < " << language_extension_.expression(batch_layout.shape().at(0)) << "; ++b) {"
           << std::endl;
    stream.setIndent(stream.indent() + 4);

    if (batch_in_memlet->base_type().primitive_type() != types::BFloat) {
        // only type different
        alloc_arr(output.main, language_extension_, batch_stride, per_batch_buf, temp_allocs);
        QuantConversion::emit_conversion(
            output,
            language_extension_,
            per_batch_src,
            batch_in_memlet->base_type(),
            per_batch_buf,
            sdfg::types::Pointer(sdfg::types::Scalar(types::BFloat)),
            sdfg::symbolic::integer(0),
            batch_stride
        );
    }

    sdfg::math::tensor::TensorLayout unbatched_layout(
        {
            symbolic::one(), // kill the outermost dimension effectively, because that is what the current qant API
                             // requires
            batch_layout.shape().at(1),
            batch_layout.shape().at(2),
            batch_layout.shape().at(3),
        },
        batch_layout.strides()
    );

    emit_dlpack_tensor_wrapper(stream, "__qant_unbatched", "E", unbatched_layout, 4);


    emit_toolkit_call(stream, result_var);

    emit_error_check(stream, result_var);

    emit_copy_result_back_and_cleanup(output, output_memlet, result_var, "B_out", out_size_str, sdfg::types::BFloat);


    for (auto& alloc : temp_allocs) {
        stream << "free(" << alloc << ");" << std::endl;
    }

    stream << per_batch_src << " += " << batch_stride_str << ";" << std::endl;

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

void BatchnormNodeDispatcher_QANT::emit_error_check(sdfg::codegen::PrettyPrinter& stream, std::string result_var) {
    stream << "if (" << result_var << " == nullptr) {" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << "throw std::runtime_error(\"QANT " << toolkit_function_name() << " n"
           << std::to_string(node_.element_id()) << " failed.\");" << std::endl;
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
    stream << std::endl;
}

void BatchnormNodeDispatcher_QANT::emit_toolkit_call(sdfg::codegen::PrettyPrinter& stream, std::string result_var) {
    stream << "DLManagedTensorVersioned* __qant_result = qant_native_computing_toolkit::batchnorm2d_fprop("
           << std::endl;
    stream << "    __qant_npu_id," << std::endl;
    stream << "    &__qant_unbatched," << std::endl;
    stream << "    &__qant_means," << std::endl;
    stream << "    &__qant_variance," << std::endl; // variances
    stream << "    &__qant_weights," << std::endl; // weights
    stream << "    &__qant_bias," << std::endl; // bias
    stream << "    epsilon" << std::endl; // eps
    stream << ");" << std::endl;
}

std::string BatchnormNodeDispatcher_QANT::toolkit_function_name() const { return "batchnorm2d_fprop"; }

} // namespace docc::qant::tensor
