#include "docc/qant/dataflow/library_nodes/math/tensor/matmul_node.h"
#include <string>

#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"
#include "sdfg/types/type.h"
#include "sdfg/types/utils.h"

namespace sdfg {
namespace math {
namespace tensor {

QantMatMulNode::QantMatMulNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const symbolic::MultiExpression& shape_a,
    const symbolic::MultiExpression& shape_b,
    const symbolic::MultiExpression& strides_a,
    const symbolic::MultiExpression& strides_b,
    symbolic::Expression offset_a,
    symbolic::Expression offset_b
)
    : MatMulNode(element_id, debug_info, vertex, parent, shape_a, shape_b, strides_a, strides_b, offset_a, offset_b) {
    code_ = LibraryNodeType_QantMatMul;
};

void QantMatMulNode::validate(const Function& function) const {
    auto& graph = this->get_parent();

    // Check that we have exactly 2 inputs and 1 output
    if (graph.in_degree(*this) != 2) {
        throw InvalidSDFGException("QantMatMulNode: Expected exactly 2 inputs (A and B)");
    }
    if (graph.out_degree(*this) != 1) {
        throw InvalidSDFGException("QantMatMulNode: Expected exactly 1 output (Y)");
    }

    // Validate K dimension matches between A and B
    auto k_a = shape_a()[shape_a().size() - 1];
    auto k_b = shape_b()[shape_b().size() - 2];
    if (!symbolic::eq(k_a, k_b)) {
        throw InvalidSDFGException(
            "QantMatMulNode: K dimension mismatch. A has K=" + k_a->__str__() + ", B has K=" + k_b->__str__()
        );
    }
}

std::string QantMatMulNode::toStr() const {
    std::stringstream ss;
    ss << "QantMatMul(";
    ss << "A=[";
    for (size_t i = 0; i < shape_a().size(); ++i) {
        if (i > 0) ss << ", ";
        ss << shape_a()[i]->__str__();
    }
    ss << "], strides_a=[";
    for (size_t i = 0; i < strides_a().size(); ++i) {
        if (i > 0) ss << ", ";
        ss << strides_a()[i]->__str__();
    }
    ss << "], offset_a=" << offset_a()->__str__();
    ss << ", B=[";
    for (size_t i = 0; i < shape_b().size(); ++i) {
        if (i > 0) ss << ", ";
        ss << shape_b()[i]->__str__();
    }
    ss << "], strides_b=[";
    for (size_t i = 0; i < strides_b().size(); ++i) {
        if (i > 0) ss << ", ";
        ss << strides_b()[i]->__str__();
    }
    ss << "], offset_b=" << offset_b()->__str__();
    ss << ")";
    return ss.str();
}


nlohmann::json QantMatMulNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const QantMatMulNode& matmul_node = static_cast<const QantMatMulNode&>(library_node);
    nlohmann::json j;

    j["code"] = matmul_node.code().value();

    serializer::JSONSerializer serializer;

    j["shape_a"] = nlohmann::json::array();
    for (auto& dim : matmul_node.shape_a()) {
        j["shape_a"].push_back(serializer.expression(dim));
    }

    j["shape_b"] = nlohmann::json::array();
    for (auto& dim : matmul_node.shape_b()) {
        j["shape_b"].push_back(serializer.expression(dim));
    }

    j["strides_a"] = nlohmann::json::array();
    for (auto& stride : matmul_node.strides_a()) {
        j["strides_a"].push_back(serializer.expression(stride));
    }

    j["strides_b"] = nlohmann::json::array();
    for (auto& stride : matmul_node.strides_b()) {
        j["strides_b"].push_back(serializer.expression(stride));
    }

    j["offset_a"] = serializer.expression(matmul_node.offset_a());
    j["offset_b"] = serializer.expression(matmul_node.offset_b());

    return j;
}

data_flow::LibraryNode& QantMatMulNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("shape_a"));
    assert(j.contains("shape_b"));

    symbolic::MultiExpression shape_a;
    for (const auto& dim : j["shape_a"]) {
        shape_a.push_back(symbolic::parse(dim.get<std::string>()));
    }

    symbolic::MultiExpression shape_b;
    for (const auto& dim : j["shape_b"]) {
        shape_b.push_back(symbolic::parse(dim.get<std::string>()));
    }

    symbolic::MultiExpression strides_a;
    if (j.contains("strides_a")) {
        for (const auto& stride : j["strides_a"]) {
            strides_a.push_back(symbolic::parse(stride.get<std::string>()));
        }
    }

    symbolic::MultiExpression strides_b;
    if (j.contains("strides_b")) {
        for (const auto& stride : j["strides_b"]) {
            strides_b.push_back(symbolic::parse(stride.get<std::string>()));
        }
    }

    symbolic::Expression offset_a = symbolic::integer(0);
    if (j.contains("offset_a")) {
        offset_a = symbolic::parse(j["offset_a"].get<std::string>());
    }

    symbolic::Expression offset_b = symbolic::integer(0);
    if (j.contains("offset_b")) {
        offset_b = symbolic::parse(j["offset_b"].get<std::string>());
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return builder
        .add_library_node<MatMulNode>(parent, debug_info, shape_a, shape_b, strides_a, strides_b, offset_a, offset_b);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
