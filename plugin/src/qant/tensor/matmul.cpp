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
    const sdfg::math::tensor::MatMulNode& node
)
    : QantTensorLibNodeDispatcher(language_extension, function, data_flow_graph, node), matmul_node_(node) {}


const sdfg::data_flow::Memlet* require_unique_output_edge(
    const sdfg::data_flow::DataFlowGraph& dflow, const sdfg::data_flow::LibraryNode& node, const std::string& conn
) {
    auto edges = dflow.out_edges_for_connector(node, conn);
    if (edges.size() != 1) {
        throw std::runtime_error("QANT MatMulNodeDispatcher_QANT: expected 1 output edge for " + conn);
    }
    return edges.at(0);
}

void MatMulNodeDispatcher_QANT::dispatch_code(
    sdfg::codegen::PrettyPrinter& stream,
    sdfg::codegen::PrettyPrinter& globals_stream,
    sdfg::codegen::CodeSnippetFactory& library_snippet_factory
) {
    QantTensorLibNodeDispatcher::emit_qant_includes_once(globals_stream, library_snippet_factory);

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
    auto& layout_a = matmul_node_.layout_a();
    auto& layout_b = matmul_node_.layout_b();

    size_t batch_dims_a = layout_a.dims() - 2;
    size_t batch_dims_b = layout_b.dims() - 2;
    size_t max_batch_dims = std::max(batch_dims_a, batch_dims_b);

    // Generate nested for-loops for batch dimensions
    std::vector<std::string> batch_vars;
    std::vector<std::string> batch_dims;
    for (size_t i = 0; i < max_batch_dims; ++i) {
        std::string var = "__qant_b" + std::to_string(i);
        batch_vars.push_back(var);

        // Determine bound for this batch dim (broadcasting: use whichever tensor has it)
        size_t a_idx = (batch_dims_a >= (max_batch_dims - i)) ? i - (max_batch_dims - batch_dims_a) : SIZE_MAX;
        size_t b_idx = (batch_dims_b >= (max_batch_dims - i)) ? i - (max_batch_dims - batch_dims_b) : SIZE_MAX;

        std::string bound;
        if (a_idx != SIZE_MAX && b_idx != SIZE_MAX) {
            auto a_dim = layout_a.shape().at(a_idx);
            auto b_dim = layout_b.shape().at(b_idx);
            auto a_bcast = symbolic::eq(a_dim, symbolic::one());
            auto b_bcast = symbolic::eq(b_dim, symbolic::one());
            if (a_bcast && b_bcast) { // both 1
                bound = "1";
            } else if (a_bcast && !b_bcast) {
                bound = language_extension_.expression(b_dim);
            } else if (!a_bcast && b_bcast) {
                bound = language_extension_.expression(a_dim);
            } else if (symbolic::eq(a_dim, b_dim)) {
                bound = language_extension_.expression(a_dim);
            } else {
                throw InvalidSDFGException("Batch Dims on n" + std::to_string(node_.element_id()) + " not broadcast!");
            }
        } else if (a_idx != SIZE_MAX) {
            bound = language_extension_.expression(layout_a.shape().at(a_idx));
        } else {
            bound = language_extension_.expression(layout_b.shape().at(b_idx));
        }

        batch_dims.push_back(bound);
        stream << "for (size_t " << var << " = 0; " << var << " < (size_t)(" << bound << "); ++" << var << ") {"
               << std::endl;
        stream.setIndent(stream.indent() + 4);
    }

    std::string a_offset = calculate_tensor_start_offset(
        layout_a.offset(), layout_a.strides(), batch_dims_a, max_batch_dims, batch_vars
    );
    std::string b_offset = calculate_tensor_start_offset(
        layout_b.offset(), layout_b.strides(), batch_dims_b, max_batch_dims, batch_vars
    );

    // Compute batch offset for Y: row-major output with shape [batch..., M, N]
    std::string y_offset = "0";
    for (size_t i = 0; i < max_batch_dims; ++i) {
        // Output stride for batch dim i = M * N * product of subsequent batch dims
        std::string c_stride = "(" + m_expr + ") * (" + n_expr + ")";
        for (size_t j = i + 1; j < max_batch_dims; ++j) {
            auto& dim = batch_dims.at(j);
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

    CodegenOutput output{.main = stream, .globals = globals_stream, .library_snippet_factory = library_snippet_factory};
    std::string a_raw_var = "__qant_A_data";
    auto required_math_type = matmul_node_.quantization();
    auto layout_a_linear = ensure_input_in_required_qant_format(
        output,
        language_extension_,
        a_src_var,
        input_a_memlet->base_type(),
        a_raw_var,
        layout_a,
        size_A,
        required_math_type,
        false,
        temp_allocs
    );

    std::string b_raw_var = "__qant_B_data";
    auto layout_b_linear = ensure_input_in_required_qant_format(
        output,
        language_extension_,
        b_src_var,
        input_b_memlet->base_type(),
        b_raw_var,
        layout_b,
        size_B,
        required_math_type,
        true,
        temp_allocs
    );

    emit_dlpack_tensor_wrapper(stream, "__qant_tensor_A", a_raw_var, layout_a_linear, 2);
    // Transposed B is always contiguous (N, K) with strides (K, 1)
    emit_dlpack_tensor_wrapper(stream, "__qant_tensor_B", b_raw_var, layout_b_linear, 2);

    stream << "DLManagedTensorVersioned* __qant_result = qant_native_computing_toolkit::linear_fprop(" << std::endl;
    stream << "    __qant_npu_id," << std::endl;
    stream << "    &__qant_tensor_A," << std::endl;
    stream << "    &__qant_tensor_B" << std::endl;
    stream << ");" << std::endl;
    stream << std::endl;

    stream << "if (__qant_result == nullptr) {" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << "throw std::runtime_error(\"QANT linear_fprop n" + std::to_string(node_.element_id()) + " failed.\");"
           << std::endl;
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;

    emit_copy_result_back_and_cleanup(output, output_memlet, "__qant_result", y_dst_var, size_C, required_math_type);

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
