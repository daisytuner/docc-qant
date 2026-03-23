#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/pooling_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"

#include "sdfg/function.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_QantPooling("ml::QantPooling");

class QantPoolingNode : public PoolingNode {
    types::PrimitiveType quantization_;

public:
    QantPoolingNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const types::PrimitiveType quantization,
        PoolingMode mode,
        const std::vector<symbolic::Expression>& shape,
        const std::vector<symbolic::Expression>& kernel_shape,
        const std::vector<symbolic::Expression>& strides,
        const std::vector<symbolic::Expression>& pads,
        const std::vector<symbolic::Expression>& dilations
    );

    void validate(const Function& function) const override;

    std::string toStr() const override;

    types::PrimitiveType quantization() const;

    void set_quantization(const types::PrimitiveType quant);
};

class QantPoolingNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
