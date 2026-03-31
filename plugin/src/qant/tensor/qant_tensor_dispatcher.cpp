#include "docc/qant/tensor/qant_tensor_dispatcher.h"

#include "daisy_rtl/primitive_types.h"
#include "docc/qant/passes/reduce_quantization_pass.h"
#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"

namespace docc::qant::tensor {

using namespace sdfg;

QantTensorLibNodeDispatcher::QantTensorLibNodeDispatcher(
    sdfg::codegen::LanguageExtension& language_extension,
    const sdfg::Function& function,
    const sdfg::data_flow::DataFlowGraph& data_flow_graph,
    const data_flow::LibraryNode& node
)
    : sdfg::codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void QantTensorLibNodeDispatcher::emit_dlpack_tensor_wrapper(
    sdfg::codegen::PrettyPrinter& stream,
    const std::string& var_name,
    const std::string& data_ptr,
    const math::tensor::TensorLayout& layout,
    int visible_dims
) {
    stream << "// Create DLPack tensor wrapper for " << var_name << std::endl;
    int dims = layout.dims();
    stream << "int64_t " << var_name << "_shape[" << visible_dims << "] = {";
    for (int i = (dims - visible_dims); i < dims; ++i) {
        stream << "static_cast<int64_t>(" << language_extension_.expression(layout.shape().at(i)) << ")";
        if (i < dims - 1) stream << ", ";
    }
    stream << "};" << std::endl;
    stream << "int64_t " << var_name << "_strides[" << visible_dims << "] = {";
    for (int i = (dims - visible_dims); i < dims; ++i) {
        stream << "static_cast<int64_t>(" << language_extension_.expression(layout.strides().at(i)) << ")";
        if (i < dims - 1) stream << ", ";
    }
    stream << "};" << std::endl;
    stream << "DLManagedTensorVersioned " << var_name << ";" << std::endl;
    stream << var_name << ".version.major = DLPACK_MAJOR_VERSION;" << std::endl;
    stream << var_name << ".version.minor = DLPACK_MINOR_VERSION;" << std::endl;
    stream << var_name << ".manager_ctx = nullptr;" << std::endl;
    stream << var_name << ".deleter = nullptr;" << std::endl;
    stream << var_name << ".flags = 0;" << std::endl;
    stream << var_name << ".dl_tensor.data = (void*)" << data_ptr << ";" << std::endl;
    stream << var_name << ".dl_tensor.device.device_type = kDLCPU;" << std::endl;
    stream << var_name << ".dl_tensor.device.device_id = 0;" << std::endl;
    stream << var_name << ".dl_tensor.ndim = " << visible_dims << ";" << std::endl;
    stream << var_name << ".dl_tensor.dtype.code = kDLBfloat;" << std::endl;
    stream << var_name << ".dl_tensor.dtype.bits = 16;" << std::endl;
    stream << var_name << ".dl_tensor.dtype.lanes = 1;" << std::endl;
    stream << var_name << ".dl_tensor.shape = " << var_name << "_shape;" << std::endl;
    stream << var_name << ".dl_tensor.strides = " << var_name << "_strides;" << std::endl;
    stream << var_name << ".dl_tensor.byte_offset = (uint64_t)(" << language_extension_.expression(layout.offset())
           << ") * sizeof(__bf16);" << std::endl;
    stream << std::endl;
}

std::string QantTensorLibNodeDispatcher::calculate_tensor_start_offset(
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

void QantTensorLibNodeDispatcher::alloc_arr(
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

sdfg::math::tensor::TensorLayout QantTensorLibNodeDispatcher::
    transposed_layout_linear(const sdfg::math::tensor::TensorLayout& layout) {
    symbolic::MultiExpression rev_shape;
    int outermost_dim = layout.dims() - 2;

    for (int i = layout.dims() - 1; i >= outermost_dim; --i) {
        rev_shape.push_back(layout.shape().at(i));
    }

    symbolic::MultiExpression strides = math::tensor::TensorLayout::linear_strides(rev_shape);
    return math::tensor::TensorLayout(rev_shape, strides, layout.offset());
}

sdfg::math::tensor::TensorLayout QantTensorLibNodeDispatcher::
    linear_layout_same_shape(const sdfg::math::tensor::TensorLayout& layout) {
    symbolic::MultiExpression same_shape;
    int outermost_dim = layout.dims() - 2;

    for (int i = outermost_dim; i < static_cast<int>(layout.dims()); ++i) {
        same_shape.push_back(layout.shape().at(i));
    }

    symbolic::MultiExpression strides = math::tensor::TensorLayout::linear_strides(same_shape);
    return math::tensor::TensorLayout(same_shape, strides, layout.offset());
}

math::tensor::TensorLayout QantTensorLibNodeDispatcher::ensure_input_in_required_qant_format(
    CodegenOutput& output,
    codegen::LanguageExtension& lang_ext,
    const std::string& src_var,
    const types::IType& input_type,
    const std::string& target_var,
    const sdfg::math::tensor::TensorLayout& layout,
    symbolic::Expression target_size,
    types::PrimitiveType target_type,
    bool require_transposed,
    std::vector<std::string>& tmp_allocs
) {
    bool transposed = layout.has_transposed_strides_no_padding();
    bool non_transposed = layout.has_linear_accesses_no_padding();

    if (!transposed && !non_transposed) {
        throw InvalidSDFGException(
            "Unsupported tensor layout on n" + std::to_string(node_.element_id()) + ": " + layout.toStr()
        );
    }

    auto inner_dim_size_str = lang_ext.expression(layout.get_dim_innermost(0));
    auto outer_dim_size_str = lang_ext.expression(layout.get_dim_innermost(1));

    // Get actual strides from source layout for correct indexing
    auto stride_outer_str = lang_ext.expression(layout.strides().at(layout.dims() - 2));
    auto stride_inner_str = lang_ext.expression(layout.strides().at(layout.dims() - 1));

    if (transposed != require_transposed) { // require a layout change

        // Convert between row-major and column-major layouts
        // linear_fprop expects:
        //   features = A (M, K) row-major, weights = B (N, K) row-major
        alloc_arr(output.main, language_extension_, target_size, target_var, tmp_allocs);
        output.main << "for (size_t __qi = 0; __qi < (size_t)(" << outer_dim_size_str << "); ++__qi) {" << std::endl;
        output.main.setIndent(output.main.indent() + 4);
        output.main << "for (size_t __qj = 0; __qj < (size_t)(" << inner_dim_size_str << "); ++__qj) {" << std::endl;
        output.main.setIndent(output.main.indent() + 4);

        // Write pattern depends on whether output should be transposed
        if (require_transposed) {
            // Output is (inner_dim, outer_dim) row-major: target[j * outer + i]
            output.main << target_var << "[__qj * (" << outer_dim_size_str << ") + __qi] = ";
        } else {
            // Output is (outer_dim, inner_dim) row-major: target[i * inner + j]
            output.main << target_var << "[__qi * (" << inner_dim_size_str << ") + __qj] = ";
        }

        bool need_conversion = input_type.primitive_type() != target_type;
        if (need_conversion) {
            output.main << "static_cast<__bf16>(";
        }
        // Read using actual strides from source (handles both row-major and column-major)
        output.main << src_var << "[__qi * (" << stride_outer_str << ") + __qj * (" << stride_inner_str << ")]";
        if (need_conversion) {
            output.main << ")";
        }
        output.main << ";" << std::endl;
        output.main.setIndent(output.main.indent() - 4);
        output.main << "}" << std::endl;
        output.main.setIndent(output.main.indent() - 4);
        output.main << "}" << std::endl;

        // Return correct layout based on require_transposed
        if (require_transposed) {
            return transposed_layout_linear(layout);
        } else {
            return linear_layout_same_shape(layout);
        }
    } else if (input_type.primitive_type() != target_type) { // only type different
        alloc_arr(output.main, language_extension_, target_size, target_var, tmp_allocs);
        QuantConversion::emit_conversion(
            output,
            language_extension_,
            src_var,
            input_type,
            target_var,
            sdfg::types::Pointer(sdfg::types::Scalar(target_type)),
            sdfg::symbolic::integer(0),
            target_size
        );
    } else { // everything matches
        output.main << "__bf16* " << target_var << " = " << src_var << ";" << std::endl;
    }

    if (transposed) { // only if we are hiding an accepted transposed input, do we need to update
        return transposed_layout_linear(layout);
    } else {
        return layout;
    }
}

} // namespace docc::qant::tensor
