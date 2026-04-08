#pragma once
#include "sdfg/data_flow/library_nodes/math/tensor/batchnorm_node.h"
#include "unary_elementwise_base_dispatcher.h"

namespace docc::qant::tensor {

class BatchnormNodeDispatcher_QANT : public QantTensorLibNodeDispatcher {
public:
    BatchnormNodeDispatcher_QANT(
        sdfg::codegen::LanguageExtension& language_extension,
        const sdfg::Function& function,
        const sdfg::data_flow::DataFlowGraph& data_flow_graph,
        const sdfg::math::tensor::BatchNormNode& node
    );

    void dispatch_code(
        sdfg::codegen::PrettyPrinter& stream,
        sdfg::codegen::PrettyPrinter& globals_stream,
        sdfg::codegen::CodeSnippetFactory& library_snippet_factory
    ) override;

protected:
    void emit_error_check(sdfg::codegen::PrettyPrinter& stream, std::string result_var);
    void emit_toolkit_call(sdfg::codegen::PrettyPrinter& stream, std::string result_var);
    std::string toolkit_function_name() const;
};

} // namespace docc::qant::tensor
