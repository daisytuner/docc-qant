#include "docc/qant/tensor/pooling.h"

#include "sdfg/data_flow/access_node.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"

#include <stdexcept>

#include "docc/qant/passes/reduce_quantization_pass.h"
#include "docc/qant/tensor/qant_tensor_dispatcher.h"

namespace docc {
namespace qant {
namespace tensor {

using namespace sdfg;

PoolingNodeDispatcher_QANT::PoolingNodeDispatcher_QANT(
    sdfg::codegen::LanguageExtension& language_extension,
    const sdfg::Function& function,
    const sdfg::data_flow::DataFlowGraph& data_flow_graph,
    const sdfg::math::tensor::PoolingNode& node
)
    : sdfg::codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node), pooling_node_(node) {}

void PoolingNodeDispatcher_QANT::emit_dlpack_tensor_wrapper_4d(
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

void PoolingNodeDispatcher_QANT::dispatch_code(
    sdfg::codegen::PrettyPrinter& stream,
    sdfg::codegen::PrettyPrinter& globals_stream,
    sdfg::codegen::CodeSnippetFactory& library_snippet_factory
) {
    QantTensorLibNodeDispatcher::emit_qant_includes_once(globals_stream, library_snippet_factory);

    auto& dflow = node_.get_parent();
    auto* input_x_memlet = dflow.in_edge_for_connector(node_, "X");
    auto out_edges = dflow.out_edges_for_connector(node_, "Y");
    if (out_edges.size() != 1) {
        throw std::runtime_error("QANT pooling dispatcher: expected exactly 1 output edge for Y");
    }
    auto* output_memlet = out_edges.at(0);

    const auto target_type = types::PrimitiveType::BFloat;

    auto& kernel_shape = pooling_node_.kernel_shape();
    if (kernel_shape.size() != 2) {
        throw std::runtime_error("QANT pooling dispatcher only supports 2D pooling.");
    }

    auto& shape = pooling_node_.shape();
    if (shape.size() != 4) {
        throw std::runtime_error("QANT pooling dispatcher requires 4D input shape [N, C, H, W].");
    }

    std::string n_expr = language_extension_.expression(shape[0]);
    std::string c_expr = language_extension_.expression(shape[1]);
    std::string h_in_expr = language_extension_.expression(shape[2]);
    std::string w_in_expr = language_extension_.expression(shape[3]);

    std::string kh_expr = language_extension_.expression(kernel_shape[0]);
    std::string kw_expr = language_extension_.expression(kernel_shape[1]);

    auto& strides = pooling_node_.strides();
    auto& pads = pooling_node_.pads();

    std::string stride_expr = strides.empty() ? "1" : language_extension_.expression(strides[0]);
    std::string padding_expr = pads.empty() ? "0" : language_extension_.expression(pads[0]);

    // Output spatial dims: H_out = (H + 2*pad - kH) / stride + 1
    std::string h_out_expr = "((" + h_in_expr + " + " + padding_expr + " + " + padding_expr + " - " + kh_expr + ") / " +
                             stride_expr + " + 1)";
    std::string w_out_expr = "((" + w_in_expr + " + " + padding_expr + " + " + padding_expr + " - " + kw_expr + ") / " +
                             stride_expr + " + 1)";

    std::string size_X = "(" + n_expr + ") * (" + c_expr + ") * (" + h_in_expr + ") * (" + w_in_expr + ")";

    auto size_X_sym = symbolic::mul(symbolic::mul(shape[0], shape[1]), symbolic::mul(shape[2], shape[3]));

    bool need_x_conversion = input_x_memlet->base_type().primitive_type() != target_type;
    bool need_y_conversion = output_memlet->base_type().primitive_type() != target_type;

    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    stream << "const uint32_t __qant_npu_id = 0;" << std::endl;
    stream << std::endl;

    std::vector<std::string> temp_allocs;

    CodegenOutput output{.main = stream, .globals = globals_stream, .library_snippet_factory = library_snippet_factory};

    // Convert X to bfloat16 if needed
    std::string x_bf16_var;
    if (need_x_conversion) {
        x_bf16_var = "__qant_X_bf16";
        stream << "__bf16* " << x_bf16_var << " = (__bf16*)malloc((" << size_X << ") * sizeof(__bf16));" << std::endl;
        temp_allocs.push_back(x_bf16_var);
        QuantConversion::emit_conversion(
            output,
            language_extension_,
            "X",
            input_x_memlet->base_type(),
            x_bf16_var,
            types::Pointer(types::Scalar(target_type)),
            symbolic::integer(0),
            size_X_sym
        );
    } else {
        x_bf16_var = "X";
    }

    // Per-sample sizes (the toolkit does not support n_batches > 1)
    std::string sample_size_X = "(" + c_expr + ") * (" + h_in_expr + ") * (" + w_in_expr + ")";
    std::string sample_size_Y = "(" + c_expr + ") * " + h_out_expr + " * " + w_out_expr;

    auto mode = pooling_node_.mode();
    bool undo_avg = (mode == math::tensor::PoolingMode::Sum);

    // Loop over the batch dimension, calling the toolkit once per sample
    stream << "for (size_t __qant_batch = 0; __qant_batch < (size_t)(" << n_expr << "); ++__qant_batch) {" << std::endl;
    stream.setIndent(stream.indent() + 4);

    // Pointer to current sample in the (possibly converted) input buffer
    stream << "__bf16* __qant_X_sample = " << x_bf16_var << " + __qant_batch * (" << sample_size_X << ");" << std::endl;

    // Create a single-sample 4D tensor wrapper [1, C, H, W]
    emit_dlpack_tensor_wrapper_4d(stream, "__qant_tensor_X", "__qant_X_sample", "1", c_expr, h_in_expr, w_in_expr);

    // Call pooling function based on mode
    stream << "DLManagedTensorVersioned* __qant_result = ";
    if (mode == math::tensor::PoolingMode::Max) {
        stream << "qant_native_computing_toolkit::maxpool2d_fprop(" << std::endl;
        stream << "    __qant_npu_id," << std::endl;
        stream << "    &__qant_tensor_X," << std::endl;
        stream << "    (size_t)(" << kh_expr << ")," << std::endl;
        stream << "    (size_t)(" << kw_expr << ")," << std::endl;
        stream << "    (size_t)(" << padding_expr << ")," << std::endl;
        stream << "    (size_t)(" << stride_expr << ")" << std::endl;
        stream << ");" << std::endl;
    } else {
        std::string count_include_pad = (mode == math::tensor::PoolingMode::Sum) ? "true" : "false";
        stream << "qant_native_computing_toolkit::avgpool2d_fprop(" << std::endl;
        stream << "    __qant_npu_id," << std::endl;
        stream << "    &__qant_tensor_X," << std::endl;
        stream << "    (size_t)(" << kh_expr << ")," << std::endl;
        stream << "    (size_t)(" << kw_expr << ")," << std::endl;
        stream << "    (size_t)(" << padding_expr << ")," << std::endl;
        stream << "    (size_t)(" << stride_expr << ")," << std::endl;
        stream << "    " << count_include_pad << std::endl;
        stream << ");" << std::endl;
    }
    stream << std::endl;

    stream << "if (__qant_result == nullptr) {" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << "throw std::runtime_error(\"QANT pooling failed.\");" << std::endl;
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;

    // Copy per-sample result into the correct offset of Y
    stream << "__bf16* __qant_result_data = reinterpret_cast<__bf16*>(__qant_result->dl_tensor.data);" << std::endl;
    stream << "size_t __qant_y_offset = __qant_batch * (" << sample_size_Y << ");" << std::endl;
    stream << "for (size_t __q_i = 0; __q_i < (size_t)(" << sample_size_Y << "); ++__q_i) {" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << "Y[__qant_y_offset + __q_i] = ";
    if (need_y_conversion) {
        stream << "static_cast<" << language_extension_.primitive_type(output_memlet->base_type().primitive_type())
               << ">(";
    }
    stream << "__qant_result_data[__q_i]";
    if (undo_avg) {
        stream << " * (__bf16)(" << kh_expr << " * " << kw_expr << ")";
    }
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

    for (auto& alloc : temp_allocs) {
        stream << "free(" << alloc << ");" << std::endl;
    }

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

} // namespace tensor
} // namespace qant
} // namespace docc
