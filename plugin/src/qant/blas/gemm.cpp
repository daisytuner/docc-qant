#include "docc/qant/blas/gemm.h"

#include "sdfg/data_flow/access_node.h"

#include <stdexcept>

namespace docc {
namespace qant {
namespace blas {

GEMMNodeDispatcher_QANT::GEMMNodeDispatcher_QANT(
    sdfg::codegen::LanguageExtension& language_extension,
    const sdfg::Function& function,
    const sdfg::data_flow::DataFlowGraph& data_flow_graph,
    const sdfg::math::blas::GEMMNode& node
)
    : sdfg::codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node), gemm_node_(node) {}

void GEMMNodeDispatcher_QANT::emit_dlpack_tensor_wrapper(
    sdfg::codegen::PrettyPrinter& stream,
    const std::string& var_name,
    const std::string& data_ptr,
    const std::string& rows,
    const std::string& cols
) {
    // Create DLManagedTensorVersioned structure on the stack
    stream << "// Create DLPack tensor wrapper for " << var_name << std::endl;
    stream << "int64_t " << var_name << "_shape[2] = {(int64_t)(" << rows << "), (int64_t)(" << cols << ")};"
           << std::endl;
    stream << "int64_t " << var_name << "_strides[2] = {(int64_t)(" << cols << "), 1};" << std::endl;
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
    stream << var_name << ".dl_tensor.dtype.lanes = 1;" << std::endl;
    stream << var_name << ".dl_tensor.shape = " << var_name << "_shape;" << std::endl;
    stream << var_name << ".dl_tensor.strides = " << var_name << "_strides;" << std::endl;
    stream << var_name << ".dl_tensor.byte_offset = 0;" << std::endl;
    stream << std::endl;
}

void GEMMNodeDispatcher_QANT::dispatch_code(
    sdfg::codegen::PrettyPrinter& stream,
    sdfg::codegen::PrettyPrinter& globals_stream,
    sdfg::codegen::CodeSnippetFactory& library_snippet_factory
) {
    // Add required includes
    globals_stream << "#include <stdfloat>" << std::endl; // Required for std::float32_t in QANT SDK
    globals_stream << "#include <dlpack/dlpack.h>" << std::endl;
    globals_stream << "#include <qant_native_computing_toolkit.h>" << std::endl;
    globals_stream << "#include <cstdlib>" << std::endl;
    globals_stream << "#include <cstring>" << std::endl;

    // Get dimensions as expressions
    std::string m_expr = language_extension_.expression(gemm_node_.m());
    std::string n_expr = language_extension_.expression(gemm_node_.n());
    std::string k_expr = language_extension_.expression(gemm_node_.k());

    // Calculate sizes
    std::string size_A = "(" + m_expr + ") * (" + k_expr + ")";
    std::string size_B = "(" + k_expr + ") * (" + n_expr + ")";
    std::string size_C = "(" + m_expr + ") * (" + n_expr + ")";

    // Check transpose requirements
    auto trans_a = gemm_node_.trans_a();
    auto trans_b = gemm_node_.trans_b();
    auto layout = gemm_node_.layout();

    if (layout == sdfg::math::blas::BLAS_Layout::ColMajor) {
        throw std::runtime_error("QANT GEMM dispatcher currently only supports RowMajor layout.");
    }

    if (trans_a != sdfg::math::blas::BLAS_Transpose::No) {
        throw std::runtime_error("QANT GEMM dispatcher requires transA = No.");
    }

    // Generate guard clause for empty dimensions
    stream << "if (" << m_expr << " != 0 && " << n_expr << " != 0 && " << k_expr << " != 0) {" << std::endl;
    stream.setIndent(stream.indent() + 4);

    stream << "const uint32_t __qant_npu_id = 0;" << std::endl;
    stream << std::endl;

    // Native bfloat16, use input directly
    if (trans_b == sdfg::math::blas::BLAS_Transpose::No) {
        // Need to transpose B
        stream << "// Transpose B from (K, N) to (N, K) for linear_fprop" << std::endl;
        stream << "__bf16* __qant_B_transposed = (__bf16*)malloc((" << size_B << ") * sizeof(__bf16));" << std::endl;
        stream << "for (size_t __qi = 0; __qi < (size_t)(" << k_expr << "); ++__qi) {" << std::endl;
        stream.setIndent(stream.indent() + 4);
        stream << "for (size_t __qj = 0; __qj < (size_t)(" << n_expr << "); ++__qj) {" << std::endl;
        stream.setIndent(stream.indent() + 4);
        stream << "__qant_B_transposed[__qj * (" << k_expr << ") + __qi] = __B[__qi * (" << n_expr << ") + __qj];"
               << std::endl;
        stream.setIndent(stream.indent() - 4);
        stream << "}" << std::endl;
        stream.setIndent(stream.indent() - 4);
        stream << "}" << std::endl;
        emit_dlpack_tensor_wrapper(stream, "__qant_tensor_A", "__A", m_expr, k_expr);
        emit_dlpack_tensor_wrapper(stream, "__qant_tensor_B", "__qant_B_transposed", n_expr, k_expr);
    } else {
        emit_dlpack_tensor_wrapper(stream, "__qant_tensor_A", "__A", m_expr, k_expr);
        emit_dlpack_tensor_wrapper(stream, "__qant_tensor_B", "__B", n_expr, k_expr);
    }

    // Call QANT linear_fprop
    stream << "// Call QANT linear_fprop: result = features @ weights^T" << std::endl;
    stream << "DLManagedTensorVersioned* __qant_result = qant_native_computing_toolkit::linear_fprop(" << std::endl;
    stream << "    __qant_npu_id," << std::endl;
    stream << "    &__qant_tensor_A," << std::endl;
    stream << "    &__qant_tensor_B" << std::endl;
    stream << ");" << std::endl;
    stream << std::endl;

    // Native bfloat16
    stream << "memcpy(__C, __qant_result->dl_tensor.data, " << size_C << " * sizeof(__bf16));" << std::endl;

    stream << std::endl;

    // Release the result tensor
    stream << "// Release QANT result tensor" << std::endl;
    stream << "if (__qant_result->deleter) {" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << "__qant_result->deleter(__qant_result);" << std::endl;
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
    stream << std::endl;

    stream << "free(__qant_B_transposed);" << std::endl;

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

} // namespace blas
} // namespace qant
} // namespace docc
