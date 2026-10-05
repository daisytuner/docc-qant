#include "docc/qant/tensor/batchnorm_dispatcher.h"

#include "sdfg/data_flow/access_node.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"

#include <stdexcept>

#include "docc/qant/passes/reduce_quantization_pass.h"
#include "docc/qant/tensor/qant_tensor_dispatcher.h"

namespace docc::qant::tensor {

using namespace sdfg;

BatchnormNodeDispatcher_QANT::BatchnormNodeDispatcher_QANT(
    sdfg::codegen::LanguageExtension& language_extension,
    const sdfg::Function& function,
    const sdfg::data_flow::DataFlowGraph& data_flow_graph,
    const sdfg::math::tensor::BatchNormNode& node
)
    : QantTensorLibNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void BatchnormNodeDispatcher_QANT::emit_dlpack_tensor_wrapper_4d(
    sdfg::codegen::PrettyPrinter& stream,
    const std::string& var_name,
    const std::string& data_ptr,
    const std::string& dim0,
    const std::string& dim1,
    const std::string& dim2,
    const std::string& dim3
) {
    stream << "// Create DLPack 4D tensor wrapper for " << var_name << std::endl;
    stream << "int64_t " << var_name << "_shape[4] = {(int64_t)(" << dim0 << "), (int64_t)(" << dim1 << "), (int64_t)("
           << dim2 << "), (int64_t)(" << dim3 << ")};" << std::endl;
    stream << "int64_t " << var_name << "_strides[4] = {"
           << "(int64_t)((" << dim1 << ") * (" << dim2 << ") * (" << dim3 << ")), "
           << "(int64_t)((" << dim2 << ") * (" << dim3 << ")), "
           << "(int64_t)(" << dim3 << "), 1};" << std::endl;
    stream << "DLManagedTensorVersioned " << var_name << ";" << std::endl;
    stream << var_name << ".version.major = DLPACK_MAJOR_VERSION;" << std::endl;
    stream << var_name << ".version.minor = DLPACK_MINOR_VERSION;" << std::endl;
    stream << var_name << ".manager_ctx = nullptr;" << std::endl;
    stream << var_name << ".deleter = nullptr;" << std::endl;
    stream << var_name << ".flags = 0;" << std::endl;
    stream << var_name << ".dl_tensor.data = (void*)" << data_ptr << ";" << std::endl;
    stream << var_name << ".dl_tensor.device.device_type = kDLCPU;" << std::endl;
    stream << var_name << ".dl_tensor.device.device_id = 0;" << std::endl;
    stream << var_name << ".dl_tensor.ndim = 4;" << std::endl;
    stream << var_name << ".dl_tensor.dtype.code = kDLBfloat;" << std::endl;
    stream << var_name << ".dl_tensor.dtype.bits = 16;" << std::endl;
    stream << var_name << ".dl_tensor.dtype.lanes = 1;" << std::endl;
    stream << var_name << ".dl_tensor.shape = " << var_name << "_shape;" << std::endl;
    stream << var_name << ".dl_tensor.strides = " << var_name << "_strides;" << std::endl;
    stream << var_name << ".dl_tensor.byte_offset = 0;" << std::endl;
    stream << std::endl;
}

void BatchnormNodeDispatcher_QANT::dispatch_code(
    sdfg::codegen::PrettyPrinter& stream,
    sdfg::codegen::PrettyPrinter& globals_stream,
    sdfg::codegen::CodeSnippetFactory& library_snippet_factory
) {
    QantTensorLibNodeDispatcher::emit_qant_includes_once(globals_stream, library_snippet_factory);

    auto& node = dynamic_cast<const sdfg::math::tensor::BatchNormNode&>(node_);
    auto& dflow = node_.get_parent();

    // Get all input memlets
    auto* batch_in_memlet = dflow.in_edge_for_connector(node_, "Batch");
    auto* means_in_memlet = dflow.in_edge_for_connector(node_, "E");
    auto* variances_in_memlet = dflow.in_edge_for_connector(node_, "Var");
    auto* gamma_in_memlet = dflow.in_edge_for_connector(node_, "Gamma");
    auto* beta_in_memlet = dflow.in_edge_for_connector(node_, "Beta");
    auto* epsilon_in_memlet = dflow.in_edge_for_connector(node_, "epsilon");
    auto* output_in_memlet = dflow.in_edge_for_connector(node_, "B_out");

    const auto target_type = types::PrimitiveType::BFloat;

    auto& batch_layout = node.batch_layout();
    if (batch_layout.dims() != 4) {
        throw std::runtime_error("QANT batchnorm dispatcher requires 4D input shape [N, C, H, W].");
    }

    std::string n_expr = language_extension_.expression(batch_layout.shape()[0]);
    std::string c_expr = language_extension_.expression(batch_layout.shape()[1]);
    std::string h_expr = language_extension_.expression(batch_layout.shape()[2]);
    std::string w_expr = language_extension_.expression(batch_layout.shape()[3]);

    std::string num_features = language_extension_.expression(node.num_features());

    // Get variable names from access nodes
    auto* batch_access = dynamic_cast<const data_flow::AccessNode*>(&batch_in_memlet->src());
    auto* means_access = dynamic_cast<const data_flow::AccessNode*>(&means_in_memlet->src());
    auto* var_access = dynamic_cast<const data_flow::AccessNode*>(&variances_in_memlet->src());
    auto* gamma_access = dynamic_cast<const data_flow::AccessNode*>(&gamma_in_memlet->src());
    auto* beta_access = dynamic_cast<const data_flow::AccessNode*>(&beta_in_memlet->src());
    auto* epsilon_access = dynamic_cast<const data_flow::AccessNode*>(&epsilon_in_memlet->src());
    auto* output_access = dynamic_cast<const data_flow::AccessNode*>(&output_in_memlet->src());

    if (!batch_access || !means_access || !var_access || !gamma_access || !beta_access || !epsilon_access ||
        !output_access) {
        throw std::runtime_error("BatchNorm dispatcher: expected AccessNode sources for all inputs");
    }

    const std::string& batch_var = batch_access->data();
    const std::string& means_var = means_access->data();
    const std::string& var_var = var_access->data();
    const std::string& gamma_var = gamma_access->data();
    const std::string& beta_var = beta_access->data();
    const std::string& epsilon_var = epsilon_access->data();
    const std::string& output_var = output_access->data();

    std::string sample_size_X = "(" + c_expr + ") * (" + h_expr + ") * (" + w_expr + ")";

    auto size_X_sym = symbolic::
        mul(symbolic::mul(batch_layout.shape()[0], batch_layout.shape()[1]),
            symbolic::mul(batch_layout.shape()[2], batch_layout.shape()[3]));

    bool need_x_conversion = batch_in_memlet->base_type().primitive_type() != target_type;
    bool need_y_conversion = output_in_memlet->base_type().primitive_type() != target_type;

    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    stream << "const uint32_t __qant_npu_id = 0;" << std::endl;
    stream << std::endl;

    std::vector<std::string> temp_allocs;

    CodegenOutput output{.main = stream, .globals = globals_stream, .library_snippet_factory = library_snippet_factory};

    // Convert 1D feature tensors to bfloat16
    stream << "// Convert feature tensors (means, variances, weights, bias) to bfloat16" << std::endl;
    stream << "__bf16* __qant_means_bf16 = (__bf16*)malloc((" << num_features << ") * sizeof(__bf16));" << std::endl;
    stream << "__bf16* __qant_var_bf16 = (__bf16*)malloc((" << num_features << ") * sizeof(__bf16));" << std::endl;
    stream << "__bf16* __qant_weights_bf16 = (__bf16*)malloc((" << num_features << ") * sizeof(__bf16));" << std::endl;
    stream << "__bf16* __qant_bias_bf16 = (__bf16*)malloc((" << num_features << ") * sizeof(__bf16));" << std::endl;
    temp_allocs.push_back("__qant_means_bf16");
    temp_allocs.push_back("__qant_var_bf16");
    temp_allocs.push_back("__qant_weights_bf16");
    temp_allocs.push_back("__qant_bias_bf16");

    stream << "for (size_t __qant_i = 0; __qant_i < (size_t)(" << num_features << "); ++__qant_i) {" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << "__qant_means_bf16[__qant_i] = static_cast<__bf16>(" << means_var << "[__qant_i]);" << std::endl;
    stream << "__qant_var_bf16[__qant_i] = static_cast<__bf16>(" << var_var << "[__qant_i]);" << std::endl;
    stream << "__qant_weights_bf16[__qant_i] = static_cast<__bf16>(" << gamma_var << "[__qant_i]);" << std::endl;
    stream << "__qant_bias_bf16[__qant_i] = static_cast<__bf16>(" << beta_var << "[__qant_i]);" << std::endl;
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
    stream << std::endl;

    // Create DLPack wrappers for 1D feature tensors
    sdfg::math::tensor::TensorLayout feature_layout({node.num_features()}, {symbolic::one()});
    emit_dlpack_tensor_wrapper(stream, "__qant_means", "__qant_means_bf16", feature_layout, 1);
    emit_dlpack_tensor_wrapper(stream, "__qant_variance", "__qant_var_bf16", feature_layout, 1);
    emit_dlpack_tensor_wrapper(stream, "__qant_weights", "__qant_weights_bf16", feature_layout, 1);
    emit_dlpack_tensor_wrapper(stream, "__qant_bias", "__qant_bias_bf16", feature_layout, 1);

    // Convert input X to bfloat16 if needed (entire batch)
    std::string x_bf16_var;
    if (need_x_conversion) {
        x_bf16_var = "__qant_X_bf16";
        std::string size_X = "(" + n_expr + ") * " + sample_size_X;
        stream << "__bf16* " << x_bf16_var << " = (__bf16*)malloc((" << size_X << ") * sizeof(__bf16));" << std::endl;
        temp_allocs.push_back(x_bf16_var);
        QuantConversion::emit_conversion(
            output,
            language_extension_,
            batch_var,
            batch_in_memlet->base_type(),
            x_bf16_var,
            types::Pointer(types::Scalar(target_type)),
            symbolic::integer(0),
            size_X_sym
        );
    } else {
        x_bf16_var = batch_var;
    }

    // Loop over the batch dimension, calling the toolkit once per sample
    stream << "// Loop over batch dimension (QANT batchnorm2d_fprop only supports batch_size=1)" << std::endl;
    stream << "for (size_t __qant_batch = 0; __qant_batch < (size_t)(" << n_expr << "); ++__qant_batch) {" << std::endl;
    stream.setIndent(stream.indent() + 4);

    // Pointer to current sample in the (possibly converted) input buffer
    stream << "__bf16* __qant_X_sample = " << x_bf16_var << " + __qant_batch * (" << sample_size_X << ");" << std::endl;

    // Create a single-sample 4D tensor wrapper [1, C, H, W]
    emit_dlpack_tensor_wrapper_4d(stream, "__qant_unbatched", "__qant_X_sample", "1", c_expr, h_expr, w_expr);

    // Create epsilon variable for toolkit call
    stream << "std::float32_t epsilon = static_cast<std::float32_t>(" << epsilon_var << ");" << std::endl;
    stream << std::endl;

    // Call batchnorm2d_fprop
    emit_toolkit_call(stream, "__qant_result");
    stream << std::endl;

    emit_error_check(stream, "__qant_result");
    stream << std::endl;

    // Copy per-sample result into the correct offset of output
    stream << "__bf16* __qant_result_data = reinterpret_cast<__bf16*>(__qant_result->dl_tensor.data);" << std::endl;
    stream << "size_t __qant_y_offset = __qant_batch * (" << sample_size_X << ");" << std::endl;
    stream << "for (size_t __q_i = 0; __q_i < (size_t)(" << sample_size_X << "); ++__q_i) {" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << output_var << "[__qant_y_offset + __q_i] = ";
    if (need_y_conversion) {
        stream << "static_cast<" << language_extension_.primitive_type(output_in_memlet->base_type().primitive_type())
               << ">(";
    }
    stream << "__qant_result_data[__q_i]";
    if (need_y_conversion) {
        stream << ")";
    }
    stream << ";" << std::endl;
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
    stream << std::endl;

    stream << "if (__qant_result->deleter) {" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << "__qant_result->deleter(__qant_result);" << std::endl;
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;

    // End batch loop
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
    stream << std::endl;

    // Free temporary allocations
    for (auto& alloc : temp_allocs) {
        stream << "free(" << alloc << ");" << std::endl;
    }

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
}

void BatchnormNodeDispatcher_QANT::emit_toolkit_call(sdfg::codegen::PrettyPrinter& stream, std::string result_var) {
    stream << "DLManagedTensorVersioned* " << result_var << " = qant_native_computing_toolkit::ai::batchnorm2d_fprop("
           << std::endl;
    stream << "    __qant_npu_id," << std::endl;
    stream << "    &__qant_unbatched," << std::endl;
    stream << "    &__qant_means," << std::endl;
    stream << "    &__qant_variance," << std::endl;
    stream << "    &__qant_weights," << std::endl;
    stream << "    &__qant_bias," << std::endl;
    stream << "    epsilon" << std::endl;
    stream << ");" << std::endl;
}

std::string BatchnormNodeDispatcher_QANT::toolkit_function_name() const { return "batchnorm2d_fprop"; }

} // namespace docc::qant::tensor
