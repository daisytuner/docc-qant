#pragma once

#include "docc/qant/tensor/qant_tensor_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_node.h"

namespace docc {
namespace qant {
namespace tensor {

/**
 * @brief Base dispatcher for all unary element-wise QANT operations.
 *
 * Handles the boilerplate that is identical across ReLU, Sigmoid, and any
 * future unary element-wise operators:
 *   - globals includes (stdfloat, dlpack, qant toolkit, …)
 *   - input/output memlet lookup for connectors "X" / "Y"
 *   - total-element-count calculation from shape
 *   - optional bf16 conversion of X
 *   - DLPack 1D tensor wrapper creation for X
 *   - result null-check, copy-back (with optional bf16→target conversion)
 *   - cleanup of the toolkit result and temporary allocations
 *
 * Subclasses only need to implement emit_toolkit_call() which emits the
 * concrete `qant_native_computing_toolkit::*_fprop(…)` invocation.
 * That method may assume:
 *   - "__qant_npu_id"    is declared as uint32_t
 *   - "__qant_tensor_X"  is a valid DLManagedTensorVersioned for the input
 * And must store the result into:
 *   - "DLManagedTensorVersioned* __qant_result"
 */
class UnaryElementWiseBaseDispatcher : public QantTensorLibNodeDispatcher {
public:
    UnaryElementWiseBaseDispatcher(
        sdfg::codegen::LanguageExtension& language_extension,
        const sdfg::Function& function,
        const sdfg::data_flow::DataFlowGraph& data_flow_graph,
        const sdfg::data_flow::LibraryNode& node,
        const sdfg::math::tensor::ElementWiseUnaryNode& ew_node
    );

    void dispatch_code(
        sdfg::codegen::PrettyPrinter& stream,
        sdfg::codegen::PrettyPrinter& globals_stream,
        sdfg::codegen::CodeSnippetFactory& library_snippet_factory
    ) override;

protected:
    /**
     * @brief Emit the operator-specific toolkit call.
     *
     * Must declare `__qant_result` as `DLManagedTensorVersioned*` and
     * assign the return value of the toolkit function to it.
     */
    virtual void emit_toolkit_call(sdfg::codegen::PrettyPrinter& stream) = 0;

    /**
     * @brief Human-readable name used in error messages (e.g. "relu_fprop").
     */
    virtual std::string toolkit_function_name() const = 0;

    const sdfg::math::tensor::ElementWiseUnaryNode& ew_node_;
};

} // namespace tensor
} // namespace qant
} // namespace docc
