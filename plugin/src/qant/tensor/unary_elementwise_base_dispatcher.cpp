#include "docc/qant/tensor/unary_elementwise_base_dispatcher.h"

#include "docc/qant/passes/reduce_quantization_pass.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/tensor.h"

#include <stdexcept>

namespace docc {
namespace qant {
namespace tensor {

using namespace sdfg;

UnaryElementWiseBaseDispatcher::UnaryElementWiseBaseDispatcher(
    sdfg::codegen::LanguageExtension& language_extension,
    const sdfg::Function& function,
    const sdfg::data_flow::DataFlowGraph& data_flow_graph,
    const sdfg::data_flow::LibraryNode& node,
    const sdfg::math::tensor::ElementWiseUnaryNode& ew_node
)
    : QantTensorLibNodeDispatcher(language_extension, function, data_flow_graph, node), ew_node_(ew_node) {}

void UnaryElementWiseBaseDispatcher::dispatch_code(
    sdfg::codegen::PrettyPrinter& stream,
    sdfg::codegen::PrettyPrinter& globals_stream,
    sdfg::codegen::CodeSnippetFactory& library_snippet_factory
) {
    QantTensorLibNodeDispatcher::emit_qant_includes_once(globals_stream, library_snippet_factory);

    auto& dflow = node_.get_parent();

    auto* input_x_memlet = dflow.in_edge_for_connector(node_, "X");
    auto* output_memlet = require_unique_output_edge(dflow, node_, "Y");

    const auto target_type = types::PrimitiveType::BFloat;

    // Calculate total number of elements from shape
    auto& shape = ew_node_.shape();
    symbolic::Expression size_expr = symbolic::one();
    std::string size_str = "1";
    for (size_t i = 0; i < shape.size(); ++i) {
        size_expr = symbolic::mul(size_expr, shape[i]);
        if (i == 0) {
            size_str = "(" + language_extension_.expression(shape[i]) + ")";
        } else {
            size_str = "(" + size_str + ") * (" + language_extension_.expression(shape[i]) + ")";
        }
    }

    bool need_x_conversion = input_x_memlet->base_type().primitive_type() != target_type;
    bool need_y_conversion = output_memlet->base_type().primitive_type() != target_type;

    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    stream << "const uint32_t __qant_npu_id = 0;" << std::endl;
    stream << "const size_t __qant_ew_size = " << size_str << ";" << std::endl;
    stream << std::endl;

    std::vector<std::string> temp_allocs;
    CodegenOutput output{.main = stream, .globals = globals_stream, .library_snippet_factory = library_snippet_factory};

    // Convert X to bfloat16 if needed
    std::string x_bf16_var;
    if (need_x_conversion) {
        x_bf16_var = "__qant_X_bf16";
        stream << "__bf16* " << x_bf16_var << " = (__bf16*)malloc(__qant_ew_size * sizeof(__bf16));" << std::endl;
        temp_allocs.push_back(x_bf16_var);
        QuantConversion::emit_conversion(
            output,
            language_extension_,
            "X",
            input_x_memlet->base_type(),
            x_bf16_var,
            types::Pointer(types::Scalar(target_type)),
            symbolic::integer(0),
            size_expr
        );
    } else {
        x_bf16_var = "X";
    }

    emit_dlpack_tensor_wrapper(
        stream, "__qant_tensor_X", x_bf16_var, math::tensor::TensorLayout({size_expr}, {symbolic::one()}), 1
    );

    // ── Operator-specific toolkit call (subclass hook) ──────────────────
    emit_toolkit_call(stream);
    stream << std::endl;

    // Null-check the result
    stream << "if (__qant_result == nullptr) {" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << "throw std::runtime_error(\"QANT " << toolkit_function_name() << " n"
           << std::to_string(node_.element_id()) << " failed.\");" << std::endl;
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
    stream << std::endl;

    emit_copy_result_back_and_cleanup(output, output_memlet, "__qant_result", "Y", size_str, types::BFloat);

    // Free temporary allocations
    for (auto& alloc : temp_allocs) {
        stream << "free(" << alloc << ");" << std::endl;
    }

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

} // namespace tensor
} // namespace qant
} // namespace docc
