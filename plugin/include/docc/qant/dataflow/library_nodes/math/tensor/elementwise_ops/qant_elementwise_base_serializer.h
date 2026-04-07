/**
 * @file elementwise_base_serializer.h
 * @brief Shared serializer for QANT elementwise library nodes (ReLU, Sigmoid, …)
 *
 * All QANT elementwise nodes share the same serialization layout (code, shape,
 * result_quant).  This serializer dispatches to the correct concrete node type
 * based on the stored code string.
 */

#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_node.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace tensor {

/**
 * @class QantElementWiseBaseSerializer
 * @brief Serializer shared by all QANT elementwise nodes
 *
 * Handles serialize/deserialize for any elementwise QANT node whose
 * persistent state is (code, shape, result_quant).
 */
template<typename T>
class QantElementWiseBaseSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json fill_base_values(const ElementWiseUnaryNode& ew_node, nlohmann::json j) {
        j["code"] = ew_node.code().value();

        serializer::JSONSerializer serializer;
        j["shape"] = nlohmann::json::array();
        for (auto& dim : ew_node.shape()) {
            j["shape"].push_back(serializer.expression(dim));
        }

        // Obtain quantization from the concrete node type.
        types::PrimitiveType quant = types::PrimitiveType::BFloat;
        auto& q_node = dynamic_cast<const T&>(ew_node);
        quant = q_node.quantization();
        j["result_quant"] = quant;

        return j;
    }

    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override {
        // All QANT elementwise nodes derive from ElementWiseUnaryNode which provides shape().
        const auto& ew_node = static_cast<const ElementWiseUnaryNode&>(library_node);
        nlohmann::json j;

        return fill_base_values(ew_node, j);
    }

    data_flow::LibraryNode& rebuild_node(
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Block& parent,
        std::vector<symbolic::Expression> shape,
        types::PrimitiveType quantization,
        DebugInfo debug_info
    ) {
        return builder.add_library_node<T>(parent, debug_info, quantization, shape);
    }

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override {
        assert(j.contains("element_id"));
        assert(j.contains("code"));
        assert(j.contains("debug_info"));
        assert(j.contains("shape"));

        serializer::JSONSerializer serializer;

        std::vector<symbolic::Expression> shape;
        for (const auto& dim : j["shape"]) {
            shape.push_back(symbolic::parse(dim.get<std::string>()));
        }

        auto result_quant = j.find("result_quant");
        types::PrimitiveType quantization = types::PrimitiveType::BFloat;
        if (result_quant != j.end()) {
            quantization = result_quant->get<types::PrimitiveType>();
        }

        DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

        std::string code = j["code"].get<std::string>();
        return rebuild_node(builder, parent, shape, quantization, debug_info);
    }
};

} // namespace tensor
} // namespace math
} // namespace sdfg
