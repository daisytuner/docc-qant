#include "docc/qant/dataflow/library_nodes/math/tensor/conv_node.h"
#include <algorithm>
#include <string>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace math {
namespace tensor {

QantConvNode::QantConvNode(
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
)
    : ConvNode(
          element_id, debug_info, vertex, parent, shape, kernel_shape, strides, pads, dilations, output_channels, group
      ),
      quantization_(quantization) {
    code_ = LibraryNodeType_QantConv;
    // Remove "B" connector inherited from ConvNode — the base dispatcher segfaults
    // on declared-but-unconnected connectors. validate() enforces that bias is not used.
    auto& ins = this->inputs();
    ins.erase(std::remove(ins.begin(), ins.end(), "B"), ins.end());
}

types::PrimitiveType QantConvNode::quantization() const { return quantization_; }

void QantConvNode::set_quantization(const types::PrimitiveType quant) { quantization_ = quant; }

void QantConvNode::validate(const Function& function) const {
    auto& graph = this->get_parent();

    // Collect input edges by connector name
    std::map<std::string, const data_flow::Memlet*> input_edges;
    for (auto& iedge : graph.in_edges(*this)) {
        input_edges[iedge.dst_conn()] = &iedge;
    }

    // Check required inputs
    if (input_edges.find("X") == input_edges.end()) {
        throw InvalidSDFGException("QantConvNode: Required input 'X' is not connected");
    }
    if (input_edges.find("W") == input_edges.end()) {
        throw InvalidSDFGException("QantConvNode: Required input 'W' is not connected");
    }

    if (graph.out_degree(*this) != 1) {
        throw InvalidSDFGException("QantConvNode: Expected exactly 1 output (Y)");
    }

    // QANT conv_fprop does not support bias
    if (input_edges.find("B") != input_edges.end()) {
        throw InvalidSDFGException("QantConvNode: Bias ('B') is not supported by QANT conv_fprop");
    }

    if (kernel_shape_.empty()) {
        throw InvalidSDFGException("QantConvNode: kernel_shape cannot be empty");
    }
}

std::string QantConvNode::toStr() const {
    std::stringstream ss;
    ss << "QantConv(";
    ss << "shape=[";
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
    ss << "], pads=[";
    for (size_t i = 0; i < pads_.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << pads_[i]->__str__();
    }
    ss << "], dilations=[";
    for (size_t i = 0; i < dilations_.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << dilations_[i]->__str__();
    }
    ss << "], output_channels=" << output_channels_->__str__();
    ss << ", group=" << group_->__str__();
    ss << ")";
    return ss.str();
}

symbolic::Expression QantConvNode::flop() const {
    // Total FLOPs = output_elements * K_conv (multiplications)
    //             + output_elements * (K_conv - 1) (additions)
    auto output_elems = num_output_elements();
    auto k_conv = kernel_iteration_count();

    auto mul_ops = symbolic::mul(output_elems, k_conv);
    auto add_ops = symbolic::mul(output_elems, symbolic::sub(k_conv, symbolic::one()));
    return symbolic::add(mul_ops, add_ops);
}

symbolic::Expression QantConvNode::num_output_elements() const {
    // N * C_out * prod(output_spatial_dim(i))
    return symbolic::mul(symbolic::mul(shape_[0], output_channels_), output_spatial_volume());
}

symbolic::Expression QantConvNode::kernel_iteration_count() const {
    // (C_in / group) * prod(kernel_shape_[i])
    return symbolic::mul(symbolic::div(shape_[1], group_), kernel_volume());
}

std::unique_ptr<data_flow::DataFlowNode> QantConvNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    auto node = std::unique_ptr<QantConvNode>(new QantConvNode(
        element_id,
        this->debug_info(),
        vertex,
        parent,
        quantization_,
        shape_,
        kernel_shape_,
        strides_,
        pads_,
        dilations_,
        output_channels_,
        group_
    ));
    node->implementation_type() = this->implementation_type();
    return node;
}

nlohmann::json QantConvNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const QantConvNode& conv_node = static_cast<const QantConvNode&>(library_node);
    nlohmann::json j;

    j["code"] = conv_node.code().value();

    serializer::JSONSerializer serializer;

    j["shape"] = nlohmann::json::array();
    for (auto& dim : conv_node.shape()) {
        j["shape"].push_back(serializer.expression(dim));
    }

    j["kernel_shape"] = nlohmann::json::array();
    for (auto& dim : conv_node.kernel_shape()) {
        j["kernel_shape"].push_back(serializer.expression(dim));
    }

    j["strides"] = nlohmann::json::array();
    for (auto& stride : conv_node.strides()) {
        j["strides"].push_back(serializer.expression(stride));
    }

    j["pads"] = nlohmann::json::array();
    for (auto& pad : conv_node.pads()) {
        j["pads"].push_back(serializer.expression(pad));
    }

    j["dilations"] = nlohmann::json::array();
    for (auto& dilation : conv_node.dilations()) {
        j["dilations"].push_back(serializer.expression(dilation));
    }

    j["output_channels"] = serializer.expression(conv_node.output_channels());
    j["group"] = serializer.expression(conv_node.group());

    j["result_quant"] = conv_node.quantization();

    return j;
}

data_flow::LibraryNode& QantConvNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("kernel_shape"));

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

    symbolic::Expression output_channels = symbolic::one();
    if (j.contains("output_channels")) {
        output_channels = symbolic::parse(j["output_channels"].get<std::string>());
    }

    symbolic::Expression group = symbolic::one();
    if (j.contains("group")) {
        group = symbolic::parse(j["group"].get<std::string>());
    }

    auto result_quant = j.find("result_quant");
    types::PrimitiveType quantization = types::BFloat;
    if (result_quant != j.end()) {
        quantization = result_quant->get<types::PrimitiveType>();
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return builder.add_library_node<QantConvNode>(
        parent, debug_info, quantization, shape, kernel_shape, strides, pads, dilations, output_channels, group
    );
}

} // namespace tensor
} // namespace math
} // namespace sdfg
