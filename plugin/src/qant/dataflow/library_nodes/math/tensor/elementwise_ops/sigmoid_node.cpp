#include "docc/qant/dataflow/library_nodes/math/tensor/elementwise_ops/sigmoid_node.h"
#include <sstream>
#include <string>

#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

QantSigmoidNode::QantSigmoidNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const types::PrimitiveType quantization,
    const std::vector<symbolic::Expression>& shape
)
    : SigmoidNode(element_id, debug_info, vertex, parent, shape), quantization_(quantization) {
    code_ = LibraryNodeType_QantSigmoid;
};

void QantSigmoidNode::validate(const Function& function) const {
    auto& graph = this->get_parent();

    // Check that we have exactly 1 input and 1 output
    if (graph.in_degree(*this) != 1) {
        throw InvalidSDFGException("QantSigmoidNode: Expected exactly 1 input (X)");
    }
    if (graph.out_degree(*this) != 1) {
        throw InvalidSDFGException("QantSigmoidNode: Expected exactly 1 output (Y)");
    }
}

std::string QantSigmoidNode::toStr() const {
    std::stringstream ss;
    ss << "QantSigmoid(";
    ss << types::primitive_type_to_string(quantization_);
    ss << ", shape: [";
    for (size_t i = 0; i < shape().size(); ++i) {
        if (i > 0) ss << ", ";
        ss << shape()[i]->__str__();
    }
    ss << "])";
    return ss.str();
}

void QantSigmoidNode::set_quantization(const types::PrimitiveType quant) { quantization_ = quant; }

symbolic::Expression QantSigmoidNode::flop() const {
    auto total_elems = SymEngine::mul(shape());
    return symbolic::mul(total_elems, symbolic::integer(2)); // per elem: 1 div, 1 exp
}

std::unique_ptr<data_flow::DataFlowNode> QantSigmoidNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    auto node = std::unique_ptr<
        QantSigmoidNode>(new QantSigmoidNode(element_id, this->debug_info(), vertex, parent, quantization_, this->shape())
    );
    node->implementation_type() = this->implementation_type();
    return node;
}


} // namespace tensor
} // namespace math
} // namespace sdfg
