#pragma once

#include "docc/qant/dataflow/library_nodes/math/tensor/qant_spatial_op_base.h"
#include "sdfg/data_flow/library_nodes/math/tensor/conv_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"

#include "sdfg/function.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_QantConv("ml::QantConv");

class QantConvNode : public ConvNode, public QantSpatialOpBase<QantConvNode> {
    types::PrimitiveType quantization_;

public:
    QantConvNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const types::PrimitiveType quantization,
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

    types::PrimitiveType quantization() const;

    void set_quantization(const types::PrimitiveType quant);

    /**
     * @brief Total number of output elements: N * C_out * prod(output_spatial_dim(i))
     */
    symbolic::Expression num_output_elements() const;

    /**
     * @brief Number of multiply-accumulate iterations per output element:
     *        (C_in / group) * prod(kernel_shape[i])
     */
    symbolic::Expression kernel_iteration_count() const;

    symbolic::Expression flop() const override;
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
