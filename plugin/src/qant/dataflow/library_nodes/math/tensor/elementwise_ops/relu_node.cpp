#include "docc/qant/dataflow/library_nodes/math/tensor/elementwise_ops/relu_node.h"
#include <sstream>
#include <string>

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


} // namespace tensor
} // namespace math
} // namespace sdfg
