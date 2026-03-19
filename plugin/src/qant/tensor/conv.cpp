#include "docc/qant/tensor/conv.h"

#include "sdfg/data_flow/access_node.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/tensor.h"

#include <stdexcept>

namespace docc {
namespace qant {
namespace tensor {

ConvNodeDispatcher_QANT::ConvNodeDispatcher_QANT(
    sdfg::codegen::LanguageExtension& language_extension,
    const sdfg::Function& function,
    const sdfg::data_flow::DataFlowGraph& data_flow_graph,
    const sdfg::math::tensor::ConvNode& node
)
    : sdfg::codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node), conv_node_(node) {}

void ConvNodeDispatcher_QANT::emit_dlpack_tensor_wrapper_4d(
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
    // Row-major strides: [dim1*dim2*dim3, dim2*dim3, dim3, 1]
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

void ConvNodeDispatcher_QANT::dispatch_code(
    sdfg::codegen::PrettyPrinter& stream,
    sdfg::codegen::PrettyPrinter& globals_stream,
    sdfg::codegen::CodeSnippetFactory& library_snippet_factory
) {
    globals_stream << "#include <stdfloat>" << std::endl;
    globals_stream << "#include <dlpack/dlpack.h>" << std::endl;
    globals_stream << "#include <qant_native_computing_toolkit.h>" << std::endl;
    globals_stream << "#include <cstdlib>" << std::endl;
    globals_stream << "#include <cstring>" << std::endl;
    globals_stream << "#include <stdexcept>" << std::endl;

    // conv_fprop only supports 2D convolution
    auto& kernel_shape = conv_node_.kernel_shape();
    if (kernel_shape.size() != 2) {
        throw std::runtime_error("QANT conv dispatcher only supports 2D convolution.");
    }

    // Input shape: [N, C_in, H, W]
    auto& shape = conv_node_.shape();
    if (shape.size() != 4) {
        throw std::runtime_error("QANT conv dispatcher requires 4D input shape [N, C_in, H, W].");
    }

    std::string n_expr = language_extension_.expression(shape[0]);
    std::string c_in_expr = language_extension_.expression(shape[1]);
    std::string h_in_expr = language_extension_.expression(shape[2]);
    std::string w_in_expr = language_extension_.expression(shape[3]);

    std::string c_out_expr = language_extension_.expression(conv_node_.output_channels());
    std::string kh_expr = language_extension_.expression(kernel_shape[0]);
    std::string kw_expr = language_extension_.expression(kernel_shape[1]);

    // Get stride, padding, dilation (conv_fprop takes uniform scalars)
    auto& strides = conv_node_.strides();
    auto& pads = conv_node_.pads();
    auto& dilations = conv_node_.dilations();

    std::string stride_expr = strides.empty() ? "1" : language_extension_.expression(strides[0]);
    std::string dilation_expr = dilations.empty() ? "1" : language_extension_.expression(dilations[0]);

    // conv_fprop takes a single padding value; use pads[0] (begin padding for H)
    std::string padding_expr = pads.empty() ? "0" : language_extension_.expression(pads[0]);

    // Compute output spatial dimensions:
    // H_out = (H_in + pad_begin + pad_end - dilation * (kH - 1) - 1) / stride + 1
    // For simplicity in size calculation, we assemble the expression directly
    std::string h_out_expr = "((" + h_in_expr + " + " + padding_expr + " + " + padding_expr + " - " + dilation_expr +
                             " * (" + kh_expr + " - 1) - 1) / " + stride_expr + " + 1)";
    std::string w_out_expr = "((" + w_in_expr + " + " + padding_expr + " + " + padding_expr + " - " + dilation_expr +
                             " * (" + kw_expr + " - 1) - 1) / " + stride_expr + " + 1)";

    std::string size_Y = "(" + n_expr + ") * (" + c_out_expr + ") * " + h_out_expr + " * " + w_out_expr;

    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    stream << "const uint32_t __qant_npu_id = 0;" << std::endl;
    stream << std::endl;

    // Features tensor: (N, C_in, H, W)
    emit_dlpack_tensor_wrapper_4d(stream, "__qant_tensor_X", "X", n_expr, c_in_expr, h_in_expr, w_in_expr);

    // Kernels tensor: (C_out, C_in, kH, kW)
    emit_dlpack_tensor_wrapper_4d(stream, "__qant_tensor_W", "W", c_out_expr, c_in_expr, kh_expr, kw_expr);

    // Call conv_fprop
    stream << "DLManagedTensorVersioned* __qant_result = qant_native_computing_toolkit::conv_fprop(" << std::endl;
    stream << "    __qant_npu_id," << std::endl;
    stream << "    &__qant_tensor_X," << std::endl;
    stream << "    &__qant_tensor_W," << std::endl;
    stream << "    (size_t)(" << padding_expr << ")," << std::endl;
    stream << "    (size_t)(" << stride_expr << ")," << std::endl;
    stream << "    (size_t)(" << dilation_expr << ")" << std::endl;
    stream << ");" << std::endl;
    stream << std::endl;

    stream << "if (__qant_result == nullptr) {" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << "throw std::runtime_error(\"QANT conv_fprop failed.\");" << std::endl;
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;

    stream << "memcpy(Y, __qant_result->dl_tensor.data, (" << size_Y << ") * sizeof(__bf16));" << std::endl;
    stream << std::endl;

    stream << "if (__qant_result->deleter) {" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << "__qant_result->deleter(__qant_result);" << std::endl;
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

} // namespace tensor
} // namespace qant
} // namespace docc
