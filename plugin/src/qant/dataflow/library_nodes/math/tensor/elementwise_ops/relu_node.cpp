#include "docc/qant/dataflow/library_nodes/math/tensor/elementwise_ops/relu_node.h"
#include <sstream>
#include <string>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

QantReLUNode::QantReLUNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const types::PrimitiveType quantization,
    const std::vector<symbolic::Expression>& shape
)
    : ReLUNode(element_id, debug_info, vertex, parent, shape), quantization_(quantization) {
    code_ = LibraryNodeType_QantReLU;
};

void QantReLUNode::validate(const Function& function) const {
    auto& graph = this->get_parent();

    // Check that we have exactly 1 input and 1 output
    if (graph.in_degree(*this) != 1) {
        throw InvalidSDFGException("QantReLUNode: Expected exactly 1 input (X)");
    }
    if (graph.out_degree(*this) != 1) {
        throw InvalidSDFGException("QantReLUNode: Expected exactly 1 output (Y)");
    }
}

std::string QantReLUNode::toStr() const {
    std::stringstream ss;
    ss << "QantReLU(";
    ss << types::primitive_type_to_string(quantization_);
    ss << ", shape: [";
    for (size_t i = 0; i < shape().size(); ++i) {
        if (i > 0) ss << ", ";
        ss << shape()[i]->__str__();
    }
    ss << "])";
    return ss.str();
}

void QantReLUNode::set_quantization(const types::PrimitiveType quant) { quantization_ = quant; }

symbolic::Expression QantReLUNode::flop() const {
    // ReLU performs one comparison operation per element: max(0, x)
    // FLOP count is the total number of elements
    symbolic::Expression total_elems = symbolic::one();
    for (const auto& dim : shape()) {
        total_elems = symbolic::mul(total_elems, dim);
    }
    return total_elems;
}

std::unique_ptr<data_flow::DataFlowNode> QantReLUNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(
        new QantReLUNode(element_id, this->debug_info(), vertex, parent, quantization_, this->shape())
    );
}

nlohmann::json QantReLUNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const QantReLUNode& relu_node = static_cast<const QantReLUNode&>(library_node);
    nlohmann::json j;

    j["code"] = relu_node.code().value();

    serializer::JSONSerializer serializer;
    j["shape"] = nlohmann::json::array();
    for (auto& dim : relu_node.shape()) {
        j["shape"].push_back(serializer.expression(dim));
    }
    j["result_quant"] = relu_node.quantization();

    return j;
}

data_flow::LibraryNode& QantReLUNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
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

    return builder.add_library_node<QantReLUNode>(parent, debug_info, quantization, shape);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
