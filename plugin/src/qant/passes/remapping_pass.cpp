#include "docc/qant/passes/remapping_pass.h"
#include "docc/qant/dataflow/library_nodes/math/tensor/conv_node.h"
#include "docc/qant/dataflow/library_nodes/math/tensor/elementwise_ops/relu_node.h"
#include "docc/qant/dataflow/library_nodes/math/tensor/matmul_node.h"
#include "docc/qant/dataflow/library_nodes/math/tensor/pooling_node.h"
#include "docc/qant/passes/remapping_pass.h"
#include "docc/qant/qant.h"
#include "docc/qant/transformations/einsum2qant_matmul.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/math/math.h"
#include "sdfg/data_flow/library_nodes/math/tensor/conv_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/relu_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/pooling_node.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/tensor.h"

#include <stdexcept>

#include "docc/qant/dataflow/library_nodes/math/tensor/elementwise_ops/sigmoid_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/batchnorm_node.h"

namespace sdfg {
namespace passes {

QantRemapping::QantRemapping(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {}

bool QantRemapping::accept(structured_control_flow::Block& node) {
    auto& dataflow = node.dataflow();

    bool made_changes = false;

    for (auto* library_node : dataflow.library_nodes()) {
        data_flow::LibraryNode* new_node = nullptr;

        if (library_node->implementation_type() != data_flow::ImplementationType_NONE &&
            library_node->implementation_type() != math::blas::ImplementationType_BLAS) {
            continue;
        }

        auto& libNode_code = library_node->code();
        if (libNode_code == math::tensor::LibraryNodeType_MatMul.value()) {
            auto* matmul_node = dynamic_cast<math::tensor::MatMulNode*>(library_node);
            auto layout_a =
                math::tensor::TensorLayout(matmul_node->shape_a(), matmul_node->strides_a(), matmul_node->offset_a());
            auto layout_b =
                math::tensor::TensorLayout(matmul_node->shape_b(), matmul_node->strides_b(), matmul_node->offset_b());
            if (!layout_a.has_linear_accesses_no_padding() && !layout_a.has_transposed_strides_no_padding() ||
                !layout_b.has_linear_accesses_no_padding() && !layout_b.has_transposed_strides_no_padding()) {
                if (report_) {
                    report_->transform_impossible("QantMatmul", "non-trivial layout");
                }
                continue; // cannot handle layouts, which we cannot classify into transposed or !transposed
            }
            auto quantization = matmul_node->primitive_type(dataflow);
            new_node = &builder_.add_library_node<
                math::tensor::QantMatMulNode>(node, matmul_node->debug_info(), quantization, layout_a, layout_b);
            new_node->implementation_type() = docc::qant::ImplementationType_QANT;
            if (report_) {
                report_->transform_applied("QantMatmul");
            }
        } else if (libNode_code == math::tensor::LibraryNodeType_Conv.value()) {
            auto* conv_node = dynamic_cast<math::tensor::ConvNode*>(library_node);

            // QANT conv_fprop does not support bias — skip if "B" is connected
            bool has_bias = false;
            for (auto& iedge : dataflow.in_edges(*conv_node)) {
                if (iedge.dst_conn() == "B") {
                    has_bias = true;
                    break;
                }
            }
            if (has_bias) {
                if (report_) {
                    report_->transform_impossible("QantConv", "has_bias");
                }
                continue;
            }

            auto quantization = conv_node->primitive_type(dataflow);
            new_node = &builder_.add_library_node<math::tensor::QantConvNode>(
                node,
                conv_node->debug_info(),
                quantization,
                conv_node->shape(),
                conv_node->kernel_shape(),
                conv_node->strides(),
                conv_node->pads(),
                conv_node->dilations(),
                conv_node->output_channels(),
                conv_node->group()
            );
            new_node->implementation_type() = docc::qant::ImplementationType_QANT;
            if (report_) {
                report_->transform_applied("QantConv");
            }
        } else if (libNode_code == math::tensor::LibraryNodeType_Pooling.value()) {
            auto* pooling_node = dynamic_cast<math::tensor::PoolingNode*>(library_node);
            auto quantization = pooling_node->primitive_type(dataflow);
            new_node = &builder_.add_library_node<math::tensor::QantPoolingNode>(
                node,
                pooling_node->debug_info(),
                quantization,
                pooling_node->mode(),
                pooling_node->shape(),
                pooling_node->kernel_shape(),
                pooling_node->strides(),
                pooling_node->pads(),
                pooling_node->dilations()
            );
            new_node->implementation_type() = docc::qant::ImplementationType_QANT;
            if (report_) {
                report_->transform_applied("QantPooling");
            }
        } else if (libNode_code == math::tensor::LibraryNodeType_ReLU.value()) {
            auto* relu_node = dynamic_cast<math::tensor::ReLUNode*>(library_node);
            auto quantization = relu_node->primitive_type(dataflow);
            new_node = &builder_.add_library_node<
                math::tensor::QantReLUNode>(node, relu_node->debug_info(), quantization, relu_node->shape());
            new_node->implementation_type() = docc::qant::ImplementationType_QANT;
            if (report_) {
                report_->transform_applied("QantReLU");
            }
        } else if (libNode_code == math::tensor::LibraryNodeType_Sigmoid.value()) {
            auto* sigmoid_node = dynamic_cast<math::tensor::SigmoidNode*>(library_node);
            auto quantization = sigmoid_node->primitive_type(dataflow);
            new_node = &builder_.add_library_node<
                math::tensor::QantSigmoidNode>(node, sigmoid_node->debug_info(), quantization, sigmoid_node->shape());
            new_node->implementation_type() = docc::qant::ImplementationType_QANT;
            if (report_) {
                report_->transform_applied("QantSigmoid");
            }
        } else if (libNode_code == sdfg::math::blas::LibraryNodeType_GEMM.value()) {
            auto* gemm_node = dynamic_cast<sdfg::math::blas::GEMMNode*>(library_node);

            gemm_node->implementation_type() = docc::qant::ImplementationType_QANT;
            made_changes = true;
            if (report_) {
                report_->transform_applied("QantGemm");
            }
        } else if (libNode_code == math::tensor::LibraryNodeType_BatchNorm.value()) {
            auto* batchnorm_node = dynamic_cast<math::tensor::BatchNormNode*>(library_node);
            auto quantization = batchnorm_node->primitive_type(dataflow);
            if (quantization == types::Float && batchnorm_node->batch_layout().dims() == 4) { // only batchnorm2d and
                                                                                              // float
                batchnorm_node->implementation_type() = docc::qant::ImplementationType_QANT;
                batchnorm_node->set_quantization(types::BFloat);
                made_changes = true;

                if (report_) {
                    report_->transform_applied("QantBatchnorm");
                }
            } else {
                if (report_) {
                    report_->transform_impossible("QantBatchnorm", "not 2d or float");
                }
            }
        } else if (library_node->code() == sdfg::einsum::LibraryNodeType_Einsum.value()) {
            auto* einsum_node = dynamic_cast<sdfg::einsum::EinsumNode*>(library_node);
            sdfg::transformations::Einsum2QantMatmul transformation(*einsum_node, "qant");
            if (transformation.can_be_applied(builder_, analysis_manager_)) {
                transformation.apply(builder_, analysis_manager_);
                made_changes = true;
                if (report_) {
                    report_->transform_applied("QantEinsum");
                }
            } else {
                if (report_) {
                    report_->transform_impossible("QantEinsum", "pattern not matched");
                }
            }
            continue;
        } else {
            continue;
        }

        if (!new_node) {
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
                node, src, *new_node, memlet.dst_conn(), memlet.subset(), pointer_type, memlet.debug_info()
            );
        }

        // Re-wire output edges
        for (auto& memlet : dataflow.out_edges(*library_node)) {
            auto& tensor_type = static_cast<const types::Tensor&>(memlet.base_type());
            types::Pointer pointer_type(tensor_type.element_type());
            auto& dst = static_cast<data_flow::AccessNode&>(memlet.dst());
            builder_.add_computational_memlet(
                node, *new_node, memlet.src_conn(), dst, memlet.subset(), pointer_type, memlet.debug_info()
            );
        }

        // Remove old edges, then old node
        for (auto* edge : old_edges) {
            builder_.remove_memlet(node, *edge);
        }
        builder_.remove_node(node, *library_node);

        made_changes = true;
    }

    return made_changes;
}

} // namespace passes
} // namespace sdfg
