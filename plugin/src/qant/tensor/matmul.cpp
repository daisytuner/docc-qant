#include "docc/qant/tensor/matmul.h"

#include "sdfg/data_flow/access_node.h"
#include "sdfg/types/tensor.h"
#include "sdfg/types/pointer.h"

#include <stdexcept>

namespace docc {
namespace qant {
namespace tensor {

MatMulNodeDispatcher_QANT::MatMulNodeDispatcher_QANT(
    sdfg::codegen::LanguageExtension& language_extension,
    const sdfg::Function& function,
    const sdfg::data_flow::DataFlowGraph& data_flow_graph,
    const sdfg::math::tensor::MatMulNode& node
)
    : sdfg::codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node), matmul_node_(node) {}

void MatMulNodeDispatcher_QANT::emit_dlpack_tensor_wrapper(
    sdfg::codegen::PrettyPrinter& stream,
    const std::string& var_name,
    const std::string& data_ptr,
    const std::string& rows,
    const std::string& cols,
    const std::string& stride_row,
    const std::string& stride_col,
    const std::string& offset
) {
    stream << "// Create DLPack tensor wrapper for " << var_name << std::endl;
    stream << "int64_t " << var_name << "_shape[2] = {(int64_t)(" << rows << "), (int64_t)(" << cols << ")};"
           << std::endl;
    stream << "int64_t " << var_name << "_strides[2] = {(int64_t)(" << stride_row << "), (int64_t)(" << stride_col << ")};" << std::endl;
    stream << "DLManagedTensorVersioned " << var_name << ";" << std::endl;
    stream << var_name << ".version.major = DLPACK_MAJOR_VERSION;" << std::endl;
    stream << var_name << ".version.minor = DLPACK_MINOR_VERSION;" << std::endl;
    stream << var_name << ".manager_ctx = nullptr;" << std::endl;
    stream << var_name << ".deleter = nullptr;" << std::endl;
    stream << var_name << ".flags = 0;" << std::endl;
    stream << var_name << ".dl_tensor.data = (void*)" << data_ptr << ";" << std::endl;
    stream << var_name << ".dl_tensor.device.device_type = kDLCPU;" << std::endl;
    stream << var_name << ".dl_tensor.device.device_id = 0;" << std::endl;
    stream << var_name << ".dl_tensor.ndim = 2;" << std::endl;
    stream << var_name << ".dl_tensor.dtype.code = kDLBfloat;" << std::endl;
    stream << var_name << ".dl_tensor.dtype.bits = 16;" << std::endl;
    stream << var_name << ".dl_tensor.dtype.lanes = 1;" << std::endl;
    stream << var_name << ".dl_tensor.shape = " << var_name << "_shape;" << std::endl;
    stream << var_name << ".dl_tensor.strides = " << var_name << "_strides;" << std::endl;
    stream << var_name << ".dl_tensor.byte_offset = (uint64_t)(" << offset << ") * sizeof(__bf16);" << std::endl;
    stream << std::endl;
}

void MatMulNodeDispatcher_QANT::dispatch_code(
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

    std::string m_expr = language_extension_.expression(matmul_node_.m());
    std::string n_expr = language_extension_.expression(matmul_node_.n());
    std::string k_expr = language_extension_.expression(matmul_node_.k());

    std::string size_B = "(" + k_expr + ") * (" + n_expr + ")";
    std::string size_C = "(" + m_expr + ") * (" + n_expr + ")";

    stream << "if (" << m_expr << " != 0 && " << n_expr << " != 0 && " << k_expr << " != 0) {" << std::endl;
    stream.setIndent(stream.indent() + 4);

    stream << "const uint32_t __qant_npu_id = 0;" << std::endl;
    stream << std::endl;

    // Compute batch dimensions (all except last 2), mirroring MatMulNode::expand()
    auto& shape_a = matmul_node_.shape_a();
    auto& shape_b = matmul_node_.shape_b();
    auto& strides_a = matmul_node_.strides_a();
    auto& strides_b = matmul_node_.strides_b();

    size_t batch_dims_a = shape_a.size() - 2;
    size_t batch_dims_b = shape_b.size() - 2;
    size_t max_batch_dims = std::max(batch_dims_a, batch_dims_b);

    // Generate nested for-loops for batch dimensions
    std::vector<std::string> batch_vars;
    for (size_t i = 0; i < max_batch_dims; ++i) {
        std::string var = "__qant_b" + std::to_string(i);
        batch_vars.push_back(var);

        // Determine bound for this batch dim (broadcasting: use whichever tensor has it)
        size_t a_idx = (batch_dims_a >= (max_batch_dims - i)) ? i - (max_batch_dims - batch_dims_a) : SIZE_MAX;
        size_t b_idx = (batch_dims_b >= (max_batch_dims - i)) ? i - (max_batch_dims - batch_dims_b) : SIZE_MAX;

        std::string bound;
        if (a_idx != SIZE_MAX) {
            bound = language_extension_.expression(shape_a[a_idx]);
        } else {
            bound = language_extension_.expression(shape_b[b_idx]);
        }

        stream << "for (size_t " << var << " = 0; " << var << " < (size_t)(" << bound << "); ++" << var << ") {" << std::endl;
        stream.setIndent(stream.indent() + 4);
    }

    // Compute batch offset for A: offset_a + sum(batch_var[i] * stride_a[i])
    std::string a_offset = language_extension_.expression(matmul_node_.offset_a());
    for (size_t i = 0; i < batch_dims_a; ++i) {
        size_t batch_idx = max_batch_dims - batch_dims_a + i;
        std::string stride = language_extension_.expression(strides_a[i]);
        a_offset = "(" + a_offset + ") + " + batch_vars[batch_idx] + " * (" + stride + ")";
    }

    // Compute batch offset for B: offset_b + sum(batch_var[i] * stride_b[i])
    std::string b_offset = language_extension_.expression(matmul_node_.offset_b());
    for (size_t i = 0; i < batch_dims_b; ++i) {
        size_t batch_idx = max_batch_dims - batch_dims_b + i;
        std::string stride = language_extension_.expression(strides_b[i]);
        b_offset = "(" + b_offset + ") + " + batch_vars[batch_idx] + " * (" + stride + ")";
    }

    // Compute batch offset for Y: row-major output with shape [batch..., M, N]
    std::string y_offset = "0";
    for (size_t i = 0; i < max_batch_dims; ++i) {
        // Output stride for batch dim i = M * N * product of subsequent batch dims
        std::string c_stride = "(" + m_expr + ") * (" + n_expr + ")";
        for (size_t j = i + 1; j < max_batch_dims; ++j) {
            size_t a_j = (batch_dims_a >= (max_batch_dims - j)) ? j - (max_batch_dims - batch_dims_a) : SIZE_MAX;
            size_t b_j = (batch_dims_b >= (max_batch_dims - j)) ? j - (max_batch_dims - batch_dims_b) : SIZE_MAX;
            std::string dim;
            if (a_j != SIZE_MAX) {
                dim = language_extension_.expression(shape_a[a_j]);
            } else {
                dim = language_extension_.expression(shape_b[b_j]);
            }
            c_stride = "(" + c_stride + ") * (" + dim + ")";
        }
        y_offset = "(" + y_offset + ") + " + batch_vars[i] + " * (" + c_stride + ")";
    }

    // Batch pointers
    stream << "__bf16* __qant_A_batch = A + (" << a_offset << ");" << std::endl;
    stream << "__bf16* __qant_B_batch = B + (" << b_offset << ");" << std::endl;
    stream << "__bf16* __qant_Y_batch = Y + (" << y_offset << ");" << std::endl;
    stream << std::endl;

    // Transpose B slice from (K, N) to (N, K) for linear_fprop
    // linear_fprop computes features @ weights^T
    // features = A (M, K), weights = B^T (N, K) → result = A @ B
    stream << "__bf16* __qant_B_transposed = (__bf16*)malloc((" << size_B << ") * sizeof(__bf16));" << std::endl;
    stream << "for (size_t __qi = 0; __qi < (size_t)(" << k_expr << "); ++__qi) {" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << "for (size_t __qj = 0; __qj < (size_t)(" << n_expr << "); ++__qj) {" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << "__qant_B_transposed[__qj * (" << k_expr << ") + __qi] = __qant_B_batch[__qi * (" << n_expr << ") + __qj];"
           << std::endl;
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;

    std::string stride_a_row = language_extension_.expression(strides_a[strides_a.size() - 2]);
    std::string stride_a_col = language_extension_.expression(strides_a[strides_a.size() - 1]);

    emit_dlpack_tensor_wrapper(stream, "__qant_tensor_A", "__qant_A_batch", m_expr, k_expr, stride_a_row, stride_a_col, "0");
    // Transposed B is always contiguous (N, K) with strides (K, 1)
    emit_dlpack_tensor_wrapper(stream, "__qant_tensor_B", "__qant_B_transposed", n_expr, k_expr, k_expr, "1", "0");

    stream << "DLManagedTensorVersioned* __qant_result = qant_native_computing_toolkit::linear_fprop(" << std::endl;
    stream << "    __qant_npu_id," << std::endl;
    stream << "    &__qant_tensor_A," << std::endl;
    stream << "    &__qant_tensor_B" << std::endl;
    stream << ");" << std::endl;
    stream << std::endl;

    stream << "if (__qant_result == nullptr) {" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << "throw std::runtime_error(\"QANT linear_fprop failed.\");" << std::endl;
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;

    stream << "memcpy(__qant_Y_batch, __qant_result->dl_tensor.data, " << size_C << " * sizeof(__bf16));" << std::endl;
    stream << std::endl;

    stream << "if (__qant_result->deleter) {" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << "__qant_result->deleter(__qant_result);" << std::endl;
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
    stream << std::endl;

    stream << "free(__qant_B_transposed);" << std::endl;

    // Close batch loops
    for (size_t i = 0; i < max_batch_dims; ++i) {
        stream.setIndent(stream.indent() - 4);
        stream << "}" << std::endl;
    }

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

} // namespace tensor
} // namespace qant
} // namespace docc
