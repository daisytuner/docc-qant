#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/conv_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"

#include "sdfg/function.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_QantConv("ml::QantConv");

class QantConvNode : public ConvNode {
public:
    QantConvNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const std::vector<symbolic::Expression>& shape,
        const std::vector<symbolic::Expression>& kernel_shape,
        const std::vector<symbolic::Expression>& strides,
        const std::vector<symbolic::Expression>& pads,
        const std::vector<symbolic::Expression>& dilations,
        symbolic::Expression output_channels,
        symbolic::Expression group
    );

    void validate(const Function& function) const override;

    std::string toStr() const override;
};

class QantConvNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
