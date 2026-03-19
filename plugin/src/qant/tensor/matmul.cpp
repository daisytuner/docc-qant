#include "docc/qant/tensor/matmul.h"

#include "sdfg/data_flow/access_node.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/tensor.h"

#include <stdexcept>

#include "docc/qant/passes/reduce_quantization_pass.h"

namespace docc {
namespace qant {
namespace tensor {

using namespace sdfg;

MatMulNodeDispatcher_QANT::MatMulNodeDispatcher_QANT(
    sdfg::codegen::LanguageExtension& language_extension,
    const sdfg::Function& function,
    const sdfg::data_flow::DataFlowGraph& data_flow_graph,
    const sdfg::math::tensor::QantMatMulNode& node
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
    stream << "int64_t " << var_name << "_strides[2] = {(int64_t)(" << stride_row << "), (int64_t)(" << stride_col
           << ")};" << std::endl;
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

std::string MatMulNodeDispatcher_QANT::calculate_tensor_start_offset(
    sdfg::symbolic::Expression tensor_offset,
    const sdfg::symbolic::MultiExpression& strides,
    size_t batch_dims,
    size_t max_batch_dims,
    std::vector<std::string> batch_vars
) {
    std::string offset = language_extension_.expression(tensor_offset);
    for (size_t i = 0; i < batch_dims; ++i) {
        size_t batch_idx = max_batch_dims - batch_dims + i;
        std::string stride = language_extension_.expression(strides[i]);
        offset = "(" + offset + ") + " + batch_vars[batch_idx] + " * (" + stride + ")";
    }
    return offset;
}


const sdfg::data_flow::Memlet* require_unique_output_edge(
    const sdfg::data_flow::DataFlowGraph& dflow, const sdfg::data_flow::LibraryNode& node, const std::string& conn
) {
    auto edges = dflow.out_edges_for_connector(node, conn);
    if (edges.size() != 1) {
        throw std::runtime_error("QANT MatMulNodeDispatcher_QANT: expected 1 output edge for " + conn);
    }
    return edges.at(0);
}

void MatMulNodeDispatcher_QANT::alloc_arr(
    sdfg::codegen::PrettyPrinter& stream,
    codegen::LanguageExtension& lang_ext,
    symbolic::Expression size,
    const std::string& var,
    std::vector<std::string>& tmp_allocs
) {
    stream << "__bf16* " << var << " = (__bf16*)malloc((" << lang_ext.expression(size) << ") * sizeof(__bf16));"
           << std::endl;
    tmp_allocs.push_back(var);
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

    auto& dflow = node_.get_parent();

    auto* input_a_memlet = dflow.in_edge_for_connector(node_, node_.input(0));
    auto* input_b_memlet = dflow.in_edge_for_connector(node_, node_.input(1));
    auto* output_memlet = require_unique_output_edge(dflow, node_, node_.output(0));


    std::string m_expr = language_extension_.expression(matmul_node_.m());
    std::string n_expr = language_extension_.expression(matmul_node_.n());
    std::string k_expr = language_extension_.expression(matmul_node_.k());

    auto size_A = symbolic::mul(matmul_node_.m(), matmul_node_.k());
    std::string size_A_str = "(" + m_expr + ") * (" + k_expr + ")";
    auto size_B = symbolic::mul(matmul_node_.k(), matmul_node_.n());
    std::string size_B_str = "(" + k_expr + ") * (" + n_expr + ")";
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

        stream << "for (size_t " << var << " = 0; " << var << " < (size_t)(" << bound << "); ++" << var << ") {"
               << std::endl;
        stream.setIndent(stream.indent() + 4);
    }

    std::string a_offset =
        calculate_tensor_start_offset(matmul_node_.offset_a(), strides_a, batch_dims_a, max_batch_dims, batch_vars);
    std::string b_offset =
        calculate_tensor_start_offset(matmul_node_.offset_b(), strides_b, batch_dims_b, max_batch_dims, batch_vars);

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

    auto a_src_var = "__raw_src_a";
    auto b_src_var = "__raw_src_b";
    auto y_dst_var = "__raw_dst";
    stream << language_extension_.declaration(a_src_var, input_a_memlet->base_type()) << " = A + (" << a_offset << ");"
           << std::endl;
    stream << language_extension_.declaration(b_src_var, input_a_memlet->base_type()) << " = B + (" << b_offset << ");"
           << std::endl;
    stream << language_extension_.declaration(y_dst_var, input_a_memlet->base_type()) << " = Y + (" << y_offset << ");"
           << std::endl;


    stream << std::endl;

    std::vector<std::string> temp_allocs;

    std::string a_raw_var;
    CodegenOutput output{.main = stream, .globals = globals_stream, .library_snippet_factory = library_snippet_factory};
    if (input_a_memlet->base_type().primitive_type() != matmul_node_.quantization()) {
        a_raw_var = "__qant_A_batch";
        alloc_arr(stream, language_extension_, size_A, a_raw_var, temp_allocs);
        QuantConversion::emit_conversion(
            output,
            language_extension_,
            a_src_var,
            input_a_memlet->base_type(),
            a_raw_var,
            sdfg::types::Pointer(sdfg::types::Scalar(matmul_node_.quantization())),
            sdfg::symbolic::integer(0),
            size_A
        );
    } else {
        a_raw_var = a_src_var;
    }

    bool transpose_b = true;
    std::string b_raw_var;
    if (transpose_b) {
        b_raw_var = "__qant_B_transposed";
        // Transpose B slice from (K, N) to (N, K) for linear_fprop
        // linear_fprop computes features @ weights^T
        // features = A (M, K), weights = B^T (N, K) → result = A @ B
        alloc_arr(stream, language_extension_, size_B, b_raw_var, temp_allocs);
        stream << "for (size_t __qi = 0; __qi < (size_t)(" << k_expr << "); ++__qi) {" << std::endl;
        stream.setIndent(stream.indent() + 4);
        stream << "for (size_t __qj = 0; __qj < (size_t)(" << n_expr << "); ++__qj) {" << std::endl;
        stream.setIndent(stream.indent() + 4);
        stream << b_raw_var << "[__qj * (" << k_expr << ") + __qi] = ";
        bool need_conversion = input_b_memlet->base_type().primitive_type() != matmul_node_.quantization();
        if (need_conversion) {
            stream << "static_cast<__bf16>(";
        }
        stream << b_src_var << "[__qi * (" << n_expr << ") + __qj]";
        if (need_conversion) {
            stream << ")";
        }
        stream << ";" << std::endl;
        stream.setIndent(stream.indent() - 4);
        stream << "}" << std::endl;
        stream.setIndent(stream.indent() - 4);
        stream << "}" << std::endl;
    } else if (input_b_memlet->base_type().primitive_type() != matmul_node_.quantization()) {
        b_raw_var = "__qant_B_batch";
        stream << "__bf16* " << b_raw_var << ";" << std::endl;
        alloc_arr(stream, language_extension_, size_B, b_raw_var, temp_allocs);
        QuantConversion::emit_conversion(
            output,
            language_extension_,
            b_src_var,
            input_b_memlet->base_type(),
            b_raw_var,
            sdfg::types::Pointer(sdfg::types::Scalar(matmul_node_.quantization())),
            sdfg::symbolic::integer(0),
            size_B
        );
    } else {
        b_raw_var = b_src_var;
    }

    std::string stride_a_row = language_extension_.expression(strides_a[strides_a.size() - 2]);
    std::string stride_a_col = language_extension_.expression(strides_a[strides_a.size() - 1]);

    emit_dlpack_tensor_wrapper(stream, "__qant_tensor_A", a_raw_var, m_expr, k_expr, stride_a_row, stride_a_col, "0");
    // Transposed B is always contiguous (N, K) with strides (K, 1)
    emit_dlpack_tensor_wrapper(stream, "__qant_tensor_B", b_raw_var, n_expr, k_expr, k_expr, "1", "0");

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

    auto need_result_conversion = output_memlet->base_type().primitive_type() != matmul_node_.quantization();
    auto y_src = "__qant_managed_result";
    stream << "__bf16* " << y_src << " = reinterpret_cast<__bf16*>(__qant_result->dl_tensor.data);" << std::endl;
    stream << "for (size_t __q_i = 0; __q_i < " << size_C << "; ++__q_i) {" << std::endl;
    stream.changeIndent(+4);
    stream << y_dst_var << "[__q_i] = ";
    if (need_result_conversion) {
        stream << "static_cast<" << language_extension_.primitive_type(output_memlet->base_type().primitive_type())
               << ">(";
    }
    stream << y_src << "[__q_i]";
    if (need_result_conversion) {
        stream << ")";
    }
    stream << ";" << std::endl;
    stream.changeIndent(-4);
    stream << "}" << std::endl;

    stream << std::endl;

    stream << "if (__qant_result->deleter) {" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << "__qant_result->deleter(__qant_result);" << std::endl;
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
    stream << std::endl;

    for (auto& alloc : temp_allocs) {
        stream << "free(" << alloc << ");" << std::endl;
    }

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
