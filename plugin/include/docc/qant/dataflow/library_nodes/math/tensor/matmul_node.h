/**
 * @file matmul_node.h
 * @brief Tensor matrix multiplication node compatible with ONNX MatMul operator
 *
 * This file defines the MatMulNode class which implements a tensor matrix
 * multiplication operation following the ONNX MatMul operator specification.
 *
 * ## ONNX MatMul Operator Compatibility
 *
 * The MatMulNode implements the ONNX MatMul operator with the following semantics:
 * - Input tensor A: [..., M, K] - arbitrary batch dimensions followed by matrix dims
 * - Input tensor B: [..., K, N] - arbitrary batch dimensions followed by matrix dims
 * - Output tensor Y: [..., M, N] - broadcasted batch dimensions with result matrix
 *
 * The operation performs matrix multiplication on the last two dimensions and
 * broadcasts over the batch dimensions following numpy broadcasting rules.
 *
 * ## Expansion
 *
 * The matmul operation is expanded into nested maps:
 * 1. Create outer maps for parallel iteration over batch and output dimensions
 * 2. Create inner loop for sequential accumulation over the K dimension
 * 3. Compute matrix multiplication using FMA (fused multiply-add) operations
 * 4. Write results to output tensor
 *
 * ## Example
 *
 * Creating a batched matrix multiplication:
 * @code
 * symbolic::MultiExpression shape_a = {symbolic::symbol("B"), symbolic::symbol("M"), symbolic::symbol("K")};
 * symbolic::MultiExpression shape_b = {symbolic::symbol("B"), symbolic::symbol("K"), symbolic::symbol("N")};
 *
 * auto& matmul_node = builder.add_library_node<math::tensor::MatMulNode>(
 *     block, debug_info, shape_a, shape_b
 * );
 * @endcode
 */

#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"

#include "sdfg/function.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_QantMatMul("ml::QantMatMul");

/**
 * @class MatMulNode
 * @brief Tensor matrix multiplication following ONNX MatMul operator specification
 *
 * MatMulNode represents a tensor matrix multiplication operation that is compatible
 * with the ONNX MatMul operator. The operation performs matrix multiplication on
 * the last two dimensions and supports broadcasting over batch dimensions.
 *
 * ## Input/Output Requirements
 * - Input connector "A": Input tensor [..., M, K]
 * - Input connector "B": Input tensor [..., K, N]
 * - Output connector "Y": Output tensor [..., M, N]
 *
 * ## Broadcasting
 * The batch dimensions are broadcast following numpy broadcasting rules:
 * - Dimensions are compared from right to left (excluding last two matrix dims)
 * - Dimensions must be equal, or one of them must be 1
 * - The output shape takes the maximum of each dimension
 *
 * ## Example
 *
 * For inputs A[B, M, K] and B[B, K, N]:
 * - Y = A @ B has shape [B, M, N]
 * - Each (b, m, n) element is computed as: sum_k(A[b, m, k] * B[b, k, n])
 */
class QantMatMulNode : public MatMulNode {
public:
    /**
     * @brief Construct a matmul node
     * @param element_id Unique element identifier
     * @param debug_info Debug information
     * @param vertex Graph vertex
     * @param parent Parent dataflow graph
     * @param shape_a Shape of input tensor A [..., M, K]
     * @param shape_b Shape of input tensor B [..., K, N]
     * @param strides_a Strides for tensor A (defaults to row-major contiguous)
     * @param strides_b Strides for tensor B (defaults to row-major contiguous)
     * @param offset_a Offset into tensor A in elements (defaults to 0)
     * @param offset_b Offset into tensor B in elements (defaults to 0)
     */
    QantMatMulNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const symbolic::MultiExpression& shape_a,
        const symbolic::MultiExpression& shape_b,
        const symbolic::MultiExpression& strides_a = {},
        const symbolic::MultiExpression& strides_b = {},
        symbolic::Expression offset_a = symbolic::integer(0),
        symbolic::Expression offset_b = symbolic::integer(0)
    );

    void validate(const Function& function) const override;

    std::string toStr() const override;
};

/**
 * @class MatMulNodeSerializer
 * @brief Serializer for MatMulNode
 */
class QantMatMulNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
