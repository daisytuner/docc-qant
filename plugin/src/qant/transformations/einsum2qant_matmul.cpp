#include "docc/qant/transformations/einsum2qant_matmul.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <string>
#include <vector>

#include "docc/qant/dataflow/library_nodes/math/tensor/matmul_node.h"
#include "docc/qant/dataflow/library_nodes/math/tensor/tensor_layout.h"
#include "docc/qant/qant.h"
#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/einsum/einsum.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/transformation.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"
#include "sdfg/types/type.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace transformations {

bool Einsum2QantMatmul::
    check_matrix_indices(long long mat, const symbolic::Symbol& indvar1, const symbolic::Symbol& indvar2) {
    // Check the last two indices of the tensor (the matrix dimensions after batch dims)
    auto& indices = this->einsum_node_.in_indices(mat);
    if (indices.size() < 2) {
        return false;
    }
    size_t last_idx = indices.size() - 1;
    size_t second_last_idx = indices.size() - 2;

    auto idx1 = this->einsum_node_.in_index(mat, second_last_idx);
    auto idx2 = this->einsum_node_.in_index(mat, last_idx);

    return !symbolic::eq(idx1, idx2) && (symbolic::eq(idx1, indvar1) || symbolic::eq(idx1, indvar2)) &&
           (symbolic::eq(idx2, indvar1) || symbolic::eq(idx2, indvar2));
}

Einsum2QantMatmul::Einsum2QantMatmul(einsum::EinsumNode& einsum_node, const std::string& target_tune)
    : einsum_node_(einsum_node), target_tune_(target_tune) {}

std::string Einsum2QantMatmul::name() const { return "Einsum2QantMatmul"; }

bool Einsum2QantMatmul::is_qant_target() const { return this->target_tune_ == "qant"; }

bool Einsum2QantMatmul::
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Check dims: must be at least 3 (for non-batched) or 4+ (for batched)
    size_t num_dims = this->einsum_node_.dims().size();
    if (num_dims < 3) {
        return false;
    }

    // Check initial values
    for (size_t i = 0; i < num_dims; i++) {
        if (!symbolic::eq(this->einsum_node_.init(i), symbolic::zero())) {
            return false;
        }
    }

    // Check out indices: should be num_dims - 1 (all except reduction dimension)
    size_t expected_out_indices = num_dims - 1;
    if (this->einsum_node_.out_indices().size() != expected_out_indices) {
        return false;
    }

    // Identify batch dimensions and matmul dimensions
    // For batched: dims = [batch..., i, j, k] where k is reduction
    // out_indices = [batch..., i, j]
    // We need to find which dimension is the reduction dimension (appears in inputs but not in output)

    // Find the reduction dimension (the one not in output)
    symbolic::Symbol indvar_inner = SymEngine::null;
    for (size_t dim_idx = 0; dim_idx < num_dims; dim_idx++) {
        auto indvar = this->einsum_node_.indvar(dim_idx);
        bool in_output = false;
        for (const auto& out_idx : this->einsum_node_.out_indices()) {
            if (symbolic::eq(out_idx, indvar)) {
                in_output = true;
                break;
            }
        }
        if (!in_output) {
            indvar_inner = indvar;
            break;
        }
    }

    if (indvar_inner.is_null()) {
        return false;
    }

    // Now identify the two matmul dimensions (non-batch, non-reduction)
    // These should be the last two dimensions in the output
    if (expected_out_indices < 2) {
        return false;
    }

    // Get the last two output indices and find their corresponding indvars
    auto out_idx_1 = this->einsum_node_.out_index(expected_out_indices - 2);
    auto out_idx_2 = this->einsum_node_.out_index(expected_out_indices - 1);

    symbolic::Symbol indvar_outer_1 = SymEngine::null;
    symbolic::Symbol indvar_outer_2 = SymEngine::null;

    for (size_t i = 0; i < num_dims; i++) {
        auto indvar = this->einsum_node_.indvar(i);
        if (symbolic::eq(indvar, out_idx_1)) {
            indvar_outer_1 = indvar;
        } else if (symbolic::eq(indvar, out_idx_2)) {
            indvar_outer_2 = indvar;
        }
    }

    if (indvar_outer_1.is_null() || indvar_outer_2.is_null()) {
        return false;
    }

    // Check bounds, i.e., prevent triangular access
    for (size_t i = 0; i < num_dims; i++) {
        auto bound = this->einsum_node_.bound(i);
        // Check that bounds don't depend on any loop variables
        for (size_t j = 0; j < num_dims; j++) {
            if (symbolic::uses(bound, this->einsum_node_.indvar(j))) {
                return false;
            }
        }
    }

    // Check inputs: expect 2 or 3 inputs (A, B, and optionally C for accumulation)
    // For now, only support 3 inputs (A, B, C/output)
    size_t num_batch_dims = expected_out_indices - 2;
    long long A = -1, B = -1, C = -1;

    if (this->einsum_node_.inputs().size() == 3) {
        C = 2;
        // Find A and B by checking which contains indvar_outer_1 and indvar_outer_2
        for (size_t i = 0; i < 2; i++) {
            auto& indices = this->einsum_node_.in_indices(i);
            if (indices.size() != expected_out_indices) {
                return false; // Inputs should match output dims (batch + 2D matrix)
            }

            // Check if this input contains indvar_outer_1 in the last 2 dims
            bool has_outer_1 = symbolic::eq(indices[indices.size() - 2], indvar_outer_1) ||
                               symbolic::eq(indices[indices.size() - 1], indvar_outer_1);
            bool has_outer_2 = symbolic::eq(indices[indices.size() - 2], indvar_outer_2) ||
                               symbolic::eq(indices[indices.size() - 1], indvar_outer_2);
            bool has_inner = symbolic::eq(indices[indices.size() - 2], indvar_inner) ||
                             symbolic::eq(indices[indices.size() - 1], indvar_inner);

            if (has_outer_1 && has_inner && !has_outer_2) {
                A = i;
            } else if (has_outer_2 && has_inner && !has_outer_1) {
                B = i;
            }
        }
    } else {
        return false; // For now, only support 3-input einsum (no alpha)
    }

    if (A == -1 || B == -1 || A == B) {
        return false;
    }

    // Verify C/output exists and has correct shape
    if (this->einsum_node_.input(C) != this->einsum_node_.output(0)) {
        return false;
    }

    // Verify batch dimensions match across all inputs
    for (size_t batch_dim = 0; batch_dim < num_batch_dims; batch_dim++) {
        auto batch_indvar = this->einsum_node_.out_index(batch_dim);

        if (!symbolic::eq(this->einsum_node_.in_index(A, batch_dim), batch_indvar) ||
            !symbolic::eq(this->einsum_node_.in_index(B, batch_dim), batch_indvar) ||
            !symbolic::eq(this->einsum_node_.in_index(C, batch_dim), batch_indvar)) {
            return false;
        }
    }

    // Check matrix indices (last 2 dims) for A and B
    size_t mat_dim_offset = num_batch_dims;
    if (!this->check_matrix_indices(A, indvar_outer_1, indvar_inner)) {
        return false;
    }
    if (!this->check_matrix_indices(B, indvar_inner, indvar_outer_2)) {
        return false;
    }

    // Get the data flow graph
    auto& dfg = this->einsum_node_.get_parent();

    // Check that we're targeting QANT
    if (!this->is_qant_target()) {
        return false;
    }

    // Determine and check the base type of output
    auto& oedge = *dfg.out_edges(this->einsum_node_).begin();
    auto data_type = oedge.base_type().primitive_type();

    // QANT supports Float, Double, and BFloat
    if (data_type != types::PrimitiveType::Float && data_type != types::PrimitiveType::Double &&
        data_type != types::PrimitiveType::BFloat) {
        return false;
    }

    // Check if all inputs have the same primitive type
    for (auto& iedge : dfg.in_edges(this->einsum_node_)) {
        if (iedge.base_type().primitive_type() != data_type) {
            return false;
        }
    }

    return true;
}

void Einsum2QantMatmul::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Get the data flow graph
    auto& dfg = this->einsum_node_.get_parent();

    // Get the block in which the einsum node lives
    auto* block = dynamic_cast<structured_control_flow::Block*>(dfg.get_parent());
    assert(block);

    // Determine the quantization type
    auto& datatype_oedge = *dfg.out_edges(this->einsum_node_).begin();
    types::PrimitiveType quantization = datatype_oedge.base_type().primitive_type();

    size_t num_dims = this->einsum_node_.dims().size();
    size_t num_out_dims = this->einsum_node_.out_indices().size();
    size_t num_batch_dims = num_out_dims - 2; // Output dims = batch dims + 2 matrix dims

    // Find the reduction dimension
    symbolic::Symbol indvar_inner = SymEngine::null;
    for (size_t dim_idx = 0; dim_idx < num_dims; dim_idx++) {
        auto indvar = this->einsum_node_.indvar(dim_idx);
        bool in_output = false;
        for (const auto& out_idx : this->einsum_node_.out_indices()) {
            if (symbolic::eq(out_idx, indvar)) {
                in_output = true;
                break;
            }
        }
        if (!in_output) {
            indvar_inner = indvar;
            break;
        }
    }
    assert(!indvar_inner.is_null());

    // Get the last two output indices (matrix dimensions) and find their corresponding indvars
    auto out_idx_1 = this->einsum_node_.out_index(num_out_dims - 2);
    auto out_idx_2 = this->einsum_node_.out_index(num_out_dims - 1);

    symbolic::Symbol indvar_outer_1 = SymEngine::null;
    symbolic::Symbol indvar_outer_2 = SymEngine::null;

    for (size_t i = 0; i < num_dims; i++) {
        auto indvar = this->einsum_node_.indvar(i);
        if (symbolic::eq(indvar, out_idx_1)) {
            indvar_outer_1 = indvar;
        } else if (symbolic::eq(indvar, out_idx_2)) {
            indvar_outer_2 = indvar;
        }
    }
    assert(!indvar_outer_1.is_null() && !indvar_outer_2.is_null());

    // Determine inputs A, B, C
    long long A = -1, B = -1, C = -1;
    if (this->einsum_node_.inputs().size() == 3) {
        C = 2;
        for (size_t i = 0; i < 2; i++) {
            auto& indices = this->einsum_node_.in_indices(i);
            size_t last = indices.size() - 1;
            size_t second_last = indices.size() - 2;

            bool has_outer_1 = symbolic::eq(this->einsum_node_.in_index(i, second_last), indvar_outer_1) ||
                               symbolic::eq(this->einsum_node_.in_index(i, last), indvar_outer_1);
            bool has_inner = symbolic::eq(this->einsum_node_.in_index(i, second_last), indvar_inner) ||
                             symbolic::eq(this->einsum_node_.in_index(i, last), indvar_inner);

            if (has_outer_1 && has_inner) {
                A = i;
            } else {
                B = i;
            }
        }
    }
    assert(A != -1 && B != -1);

    // Get actual tensor shapes from the input edges instead of symbolic bounds
    auto* input_a_edge = dfg.in_edge_for_connector(this->einsum_node_, this->einsum_node_.input(A));
    auto* input_b_edge = dfg.in_edge_for_connector(this->einsum_node_, this->einsum_node_.input(B));

    auto& tensor_a_type = static_cast<const types::Tensor&>(input_a_edge->base_type());
    auto& tensor_b_type = static_cast<const types::Tensor&>(input_b_edge->base_type());

    auto& shape_a_actual = tensor_a_type.shape();
    auto& shape_b_actual = tensor_b_type.shape();

    // Build tensor shapes and strides using actual tensor dimensions
    symbolic::MultiExpression shape_a, shape_b, strides_a, strides_b;

    // For shape_a and shape_b, we can directly use the actual tensor shapes
    // since the transformation validates that the indices match the matmul pattern

    // Copy shapes from actual tensor types
    shape_a = shape_a_actual;
    shape_b = shape_b_actual;

    // Compute row-major strides
    strides_a = math::tensor::TensorLayout::linear_strides(shape_a);
    strides_b = math::tensor::TensorLayout::linear_strides(shape_b);

    // Create tensor layouts
    math::tensor::TensorLayout layout_a(shape_a, strides_a, symbolic::integer(0));
    math::tensor::TensorLayout layout_b(shape_b, strides_b, symbolic::integer(0));

    // Add the QantMatMul node
    auto& libnode = builder.add_library_node<
        math::tensor::QantMatMulNode>(*block, this->einsum_node_.debug_info(), quantization, layout_a, layout_b);

    // Set the implementation type to QANT
    libnode.implementation_type() = docc::qant::ImplementationType_QANT;

    // Copy the memlets - QantMatMulNode uses connectors "A", "B", "Y"
    // Convert Tensor types to Pointer types for computational memlets
    // Note: We ignore alpha if it exists - QantMatMul doesn't support scalar multiplication
    for (auto& iedge : dfg.in_edges(this->einsum_node_)) {
        if (iedge.dst_conn() == this->einsum_node_.input(A)) {
            auto& tensor_type = static_cast<const types::Tensor&>(iedge.base_type());
            types::Pointer pointer_type(tensor_type.element_type());
            builder.add_memlet(
                *block, iedge.src(), iedge.src_conn(), libnode, "A", iedge.subset(), pointer_type, iedge.debug_info()
            );
        } else if (iedge.dst_conn() == this->einsum_node_.input(B)) {
            auto& tensor_type = static_cast<const types::Tensor&>(iedge.base_type());
            types::Pointer pointer_type(tensor_type.element_type());
            builder.add_memlet(
                *block, iedge.src(), iedge.src_conn(), libnode, "B", iedge.subset(), pointer_type, iedge.debug_info()
            );
        }
        // Skip C input and alpha - QantMatMul handles accumulation internally
    }

    for (auto& oedge : dfg.out_edges(this->einsum_node_)) {
        if (oedge.src_conn() == this->einsum_node_.output(0)) {
            auto& tensor_type = static_cast<const types::Tensor&>(oedge.base_type());
            types::Pointer pointer_type(tensor_type.element_type());
            builder.add_memlet(
                *block, libnode, "Y", oedge.dst(), oedge.dst_conn(), oedge.subset(), pointer_type, oedge.debug_info()
            );
        }
    }

    // Remove the old memlets
    while (dfg.in_edges(this->einsum_node_).begin() != dfg.in_edges(this->einsum_node_).end()) {
        auto& iedge = *dfg.in_edges(this->einsum_node_).begin();
        builder.remove_memlet(*block, iedge);
    }
    while (dfg.out_edges(this->einsum_node_).begin() != dfg.out_edges(this->einsum_node_).end()) {
        auto& oedge = *dfg.out_edges(this->einsum_node_).begin();
        builder.remove_memlet(*block, oedge);
    }

    // Remove the einsum node
    builder.remove_node(*block, this->einsum_node_);

    analysis_manager.invalidate_all();
}

void Einsum2QantMatmul::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["einsum_node_element_id"] = this->einsum_node_.element_id();
    j["target_tune"] = this->target_tune_;
}

Einsum2QantMatmul Einsum2QantMatmul::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    assert(j.contains("einsum_node_element_id"));
    assert(j["einsum_node_element_id"].is_number_unsigned());
    assert(j.contains("impl_type"));

    size_t einsum_node_id = j["einsum_node_element_id"].get<size_t>();
    auto* einsum_node_element = builder.find_element_by_id(einsum_node_id);
    if (!einsum_node_element) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(einsum_node_id) + " not found"
        );
    }
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(einsum_node_element);
    if (!einsum_node) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(einsum_node_id) + " is not an EinsumNode"
        );
    }

    std::string target_tune;
    if (j.contains("target_tune")) {
        target_tune = j.at("target_tune").get<std::string>();
    } else {
        target_tune = "none";
    }

    return Einsum2QantMatmul(*einsum_node, target_tune);
}

} // namespace transformations
} // namespace sdfg
