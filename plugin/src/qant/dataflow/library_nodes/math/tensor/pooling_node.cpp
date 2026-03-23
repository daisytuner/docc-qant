#include "docc/qant/dataflow/library_nodes/math/tensor/pooling_node.h"
#include <string>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace math {
namespace tensor {

QantPoolingNode::QantPoolingNode(
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
)
    : PoolingNode(element_id, debug_info, vertex, parent, mode, shape, kernel_shape, strides, pads, dilations),
      quantization_(quantization) {
    code_ = LibraryNodeType_QantPooling;
}

types::PrimitiveType QantPoolingNode::quantization() const { return quantization_; }

void QantPoolingNode::set_quantization(const types::PrimitiveType quant) { quantization_ = quant; }

void QantPoolingNode::validate(const Function& function) const {
    auto& graph = this->get_parent();

    if (graph.in_degree(*this) != 1) {
        throw InvalidSDFGException("QantPoolingNode: Expected exactly 1 input (X)");
    }
    if (graph.out_degree(*this) != 1) {
        throw InvalidSDFGException("QantPoolingNode: Expected exactly 1 output (Y)");
    }

    if (kernel_shape_.empty()) {
        throw InvalidSDFGException("QantPoolingNode: kernel_shape cannot be empty");
    }

    if (kernel_shape_.size() != 2) {
        throw InvalidSDFGException("QantPoolingNode: only 2D pooling is supported");
    }

    if (shape_.size() != 4) {
        throw InvalidSDFGException("QantPoolingNode: requires 4D input shape [N, C, H, W]");
    }
}

std::string QantPoolingNode::toStr() const {
    std::stringstream ss;
    ss << "QantPooling(mode=" << PoolingNode::mode_to_string(mode_);
    ss << ", shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << shape_[i]->__str__();
    }
    ss << "], kernel_shape=[";
    for (size_t i = 0; i < kernel_shape_.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << kernel_shape_[i]->__str__();
    }
    ss << "], strides=[";
    for (size_t i = 0; i < strides_.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << strides_[i]->__str__();
    }
    ss << "])";
    return ss.str();
}

nlohmann::json QantPoolingNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const QantPoolingNode& node = static_cast<const QantPoolingNode&>(library_node);
    nlohmann::json j;

    j["code"] = node.code().value();
    j["mode"] = PoolingNode::mode_to_string(node.mode());

    serializer::JSONSerializer serializer;

    j["shape"] = nlohmann::json::array();
    for (auto& dim : node.shape()) {
        j["shape"].push_back(serializer.expression(dim));
    }

    j["kernel_shape"] = nlohmann::json::array();
    for (auto& dim : node.kernel_shape()) {
        j["kernel_shape"].push_back(serializer.expression(dim));
    }

    j["strides"] = nlohmann::json::array();
    for (auto& stride : node.strides()) {
        j["strides"].push_back(serializer.expression(stride));
    }

    j["pads"] = nlohmann::json::array();
    for (auto& pad : node.pads()) {
        j["pads"].push_back(serializer.expression(pad));
    }

    j["dilations"] = nlohmann::json::array();
    for (auto& dilation : node.dilations()) {
        j["dilations"].push_back(serializer.expression(dilation));
    }

    j["result_quant"] = node.quantization();

    return j;
}

data_flow::LibraryNode& QantPoolingNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("mode"));
    assert(j.contains("kernel_shape"));

    auto mode = PoolingNode::string_to_mode(j["mode"].get<std::string>());

    std::vector<symbolic::Expression> shape;
    if (j.contains("shape")) {
        for (const auto& dim : j["shape"]) {
            shape.push_back(symbolic::parse(dim.get<std::string>()));
        }
    }

    std::vector<symbolic::Expression> kernel_shape;
    for (const auto& dim : j["kernel_shape"]) {
        kernel_shape.push_back(symbolic::parse(dim.get<std::string>()));
    }

    std::vector<symbolic::Expression> strides;
    if (j.contains("strides")) {
        for (const auto& stride : j["strides"]) {
            strides.push_back(symbolic::parse(stride.get<std::string>()));
        }
    }

    std::vector<symbolic::Expression> pads;
    if (j.contains("pads")) {
        for (const auto& pad : j["pads"]) {
            pads.push_back(symbolic::parse(pad.get<std::string>()));
        }
    }

    std::vector<symbolic::Expression> dilations;
    if (j.contains("dilations")) {
        for (const auto& dilation : j["dilations"]) {
            dilations.push_back(symbolic::parse(dilation.get<std::string>()));
        }
    }

    auto result_quant = j.find("result_quant");
    types::PrimitiveType quantization = types::BFloat;
    if (result_quant != j.end()) {
        quantization = result_quant->get<types::PrimitiveType>();
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return builder.add_library_node<
        QantPoolingNode>(parent, debug_info, quantization, mode, shape, kernel_shape, strides, pads, dilations);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
