#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/sigmoid_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"

#include "sdfg/function.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_QantSigmoid("ml::QantSigmoid");

class QantSigmoidNode : public SigmoidNode {
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
    QantSigmoidNode(
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

    void set_quantization(const types::PrimitiveType quant);

    symbolic::Expression flop() const override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;
};


} // namespace tensor
} // namespace math
} // namespace sdfg
