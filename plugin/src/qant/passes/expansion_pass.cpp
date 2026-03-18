#include "docc/qant/passes/expansion_pass.h"
#include "docc/qant/qant.h"
#include "docc/qant/dataflow/library_nodes/math/tensor/matmul_node.h"

#include "sdfg/data_flow/library_nodes/math/math.h"
#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/types/tensor.h"
#include "sdfg/types/pointer.h"

namespace sdfg {
namespace passes {

QantExpansion::QantExpansion(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::StructuredSDFGVisitor(builder, analysis_manager) {}

bool QantExpansion::accept(structured_control_flow::Block& node) {
    auto& dataflow = node.dataflow();

    for (auto* library_node : dataflow.library_nodes()) {

        if (library_node->implementation_type() == docc::qant::ImplementationType_TensorQANT) {
            data_flow::LibraryNode* new_node = nullptr;

            if (library_node->code() == math::tensor::LibraryNodeType_MatMul.value()) {
                auto* matmul_node = dynamic_cast<math::tensor::MatMulNode*>(library_node);
                new_node = &builder_.add_library_node<math::tensor::QantMatMulNode>(
                    node,
                    matmul_node->debug_info(),
                    matmul_node->shape_a(),
                    matmul_node->shape_b(),
                    matmul_node->strides_a(),
                    matmul_node->strides_b(),
                    matmul_node->offset_a(),
                    matmul_node->offset_b()
                );
                new_node->implementation_type() = docc::qant::ImplementationType_QANT;
            }

            // Expand math nodes that have no mapping
            if (!new_node) {
                if (auto math_node = dynamic_cast<math::MathNode*>(library_node)) {
                    if (math_node->expand(this->builder_, this->analysis_manager_)) {
                        return true;
                    }
                }
                continue;
            }

            // Collect old edges before modifying the graph
            std::vector<const data_flow::Memlet*> old_edges;
            for (auto& memlet : dataflow.in_edges(*library_node)) {
                old_edges.push_back(&memlet);
            }
            for (auto& memlet : dataflow.out_edges(*library_node)) {
                old_edges.push_back(&memlet);
            }

            // Re-wire input edges
            for (auto& memlet : dataflow.in_edges(*library_node)) {
                auto& tensor_type = static_cast<const types::Tensor&>(memlet.base_type());
                types::Pointer pointer_type(tensor_type.element_type());
                auto& src = static_cast<data_flow::AccessNode&>(memlet.src());
                builder_.add_computational_memlet(
                    node, src, *new_node, memlet.dst_conn(),
                    memlet.subset(), pointer_type, memlet.debug_info()
                );
            }

            // Re-wire output edges
            for (auto& memlet : dataflow.out_edges(*library_node)) {
                auto& tensor_type = static_cast<const types::Tensor&>(memlet.base_type());
                types::Pointer pointer_type(tensor_type.element_type());
                auto& dst = static_cast<data_flow::AccessNode&>(memlet.dst());
                builder_.add_computational_memlet(
                    node, *new_node, memlet.src_conn(), dst,
                    memlet.subset(), pointer_type, memlet.debug_info()
                );
            }

            // Remove old edges, then old node (keep access nodes — they're connected to new_node)
            for (auto* edge : old_edges) {
                builder_.remove_memlet(node, *edge);
            }
            builder_.remove_node(node, *library_node);

            return true;
        }
        else if (library_node->implementation_type() != data_flow::ImplementationType_NONE) {
            continue;
        }

        if (auto math_node = dynamic_cast<math::MathNode*>(library_node)) {
            if (math_node->expand(this->builder_, this->analysis_manager_)) {
                return true;
            }
        }

    }
    return false;
}

} // namespace passes
} // namespace sdfg
