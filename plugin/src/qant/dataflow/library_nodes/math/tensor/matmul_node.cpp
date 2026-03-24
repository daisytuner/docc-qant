#include "docc/qant/dataflow/library_nodes/math/tensor/matmul_node.h"
#include <string>

#include "daisy_rtl/primitive_types.h"
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
    const types::PrimitiveType quantization,
    const TensorLayout& layout_a,
    const TensorLayout& layout_b
)
    : MatMulNode(
          element_id,
          debug_info,
          vertex,
          parent,
          layout_a.shape(),
          layout_b.shape(),
          layout_a.strides(),
          layout_b.strides(),
          layout_a.offset(),
          layout_b.offset()
      ),
      quantization_(quantization), layout_a_(layout_a), layout_b_(layout_b) {
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
    ss << types::primitive_type_to_string(quantization_) << ", ";
    ss << "A: " << layout_a_;
    ss << ", B: " << layout_b_;

    ss << ")";
    return ss.str();
}

/**
 * Warning: this is wrong, as it assumes linear iteration. If the strides were used to skip over some element, this
 * reduction will not be reflected in the current flop estimation. For this, fix the m, n, k accessors that only rely on
 * shape
 */
symbolic::Expression QantMatMulNode::flop() const {
    auto res_elems = symbolic::mul(this->m(), this->n());
    auto k = this->k();

    auto mm_mul_ops = symbolic::mul(res_elems, k);
    auto mm_sum_ops = symbolic::mul(res_elems, symbolic::sub(k, symbolic::one()));

    auto mul_ops = mm_mul_ops;
    auto add_ops = mm_sum_ops;
    auto per_mat = symbolic::add(mul_ops, add_ops);
    int a_dims = layout_a_.dims();
    int b_dims = layout_b_.dims();
    if (a_dims > 2 || b_dims > 2) {
        std::vector<symbolic::Expression> factors{per_mat};
        auto max_dims = std::max(a_dims, b_dims);
        for (int i = 2; i < max_dims; ++i) {
            symbolic::Expression dim_a, dim_b;
            if (i < a_dims) {
                dim_a = layout_a_.get_dim_innermost(i);
            }
            if (i < b_dims) {
                dim_b = layout_b_.get_dim_innermost(i);
            }
            if (dim_a.is_null() & !dim_b.is_null()) {
                factors.push_back(dim_b);
            } else if (!dim_a.is_null() & dim_b.is_null()) {
                factors.push_back(dim_a);
            } else if (!dim_a.is_null() & !dim_b.is_null()) {
                if (!symbolic::eq(dim_a, dim_b)) {
                    throw InvalidSDFGException(
                        "Batch dimension " + std::to_string(i) + " mismatch between A and B. A has " +
                        dim_a->__str__() + ", B has " + dim_b->__str__()
                    );
                } else {
                    factors.push_back(dim_a);
                }
            } else {
                return SymEngine::null;
            }
        }
        return SymEngine::mul(factors);
    } else {
        return per_mat;
    }
}

types::PrimitiveType QantMatMulNode::quantization() const { return quantization_; }

void QantMatMulNode::set_quantization(const types::PrimitiveType quant) { quantization_ = quant; }

const TensorLayout& QantMatMulNode::layout_a() const { return layout_a_; }
const TensorLayout& QantMatMulNode::layout_b() const { return layout_b_; }


nlohmann::json QantMatMulNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const QantMatMulNode& matmul_node = static_cast<const QantMatMulNode&>(library_node);
    nlohmann::json j;

    j["code"] = matmul_node.code().value();

    serializer::JSONSerializer serializer;

    matmul_node.layout_a().serialize_to_json(j["layout_a"]);
    matmul_node.layout_b().serialize_to_json(j["layout_b"]);

    j["result_quant"] = matmul_node.quantization();

    return j;
}

data_flow::LibraryNode& QantMatMulNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));

    auto layout_a = TensorLayout::deserialize_from_json(j.at("layout_a"));
    auto layout_b = TensorLayout::deserialize_from_json(j.at("layout_b"));

    auto result_quant = j.find("result_quant");
    types::PrimitiveType quantization = types::BFloat;
    if (result_quant != j.end()) {
        quantization = result_quant->get<types::PrimitiveType>();
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return builder.add_library_node<QantMatMulNode>(parent, debug_info, quantization, layout_a, layout_b);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
