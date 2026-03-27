/**
 * @file relu_node.h
 * @brief ReLU activation node for QANT backend
 *
 * This file defines the QantReLUNode class which implements a ReLU activation
 * operation for the QANT accelerator.
 */

#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/relu_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"

#include "sdfg/function.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_QantReLU("ml::QantReLU");

/**
 * @class QantReLUNode
 * @brief ReLU activation node for QANT backend
 *
 * QantReLUNode represents a ReLU activation operation that is executed on
 * the QANT accelerator using the qant_native_computing_toolkit::relu_fprop API.
 *
 * ## Input/Output Requirements
 * - Input connector "X": Input tensor of any shape
 * - Output connector "Y": Output tensor with same shape as input
 *
 * The operation computes: Y = max(0, X) element-wise
 */
class QantReLUNode : public ReLUNode {
    types::PrimitiveType quantization_;

public:
    /**
     * @brief Construct a QANT ReLU node
     * @param element_id Unique element identifier
     * @param debug_info Debug information
     * @param vertex Graph vertex
     * @param parent Parent dataflow graph
     * @param quantization Primitive type for computation (typically BFloat16)
     * @param shape Shape of the input/output tensor
     */
    QantReLUNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const types::PrimitiveType quantization,
        const std::vector<symbolic::Expression>& shape
    );

    void validate(const Function& function) const override;
    std::string toStr() const override;

    types::PrimitiveType quantization() const { return quantization_; }

    symbolic::Expression flop() const override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;
};

/**
 * @class QantReLUNodeSerializer
 * @brief Serializer for QantReLUNode
 */
class QantReLUNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
