#pragma once

#include "docc/qant/qant.h"
#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"

namespace docc {
namespace qant {
namespace blas {

class GEMMNodeDispatcher_QANT : public sdfg::codegen::LibraryNodeDispatcher {
private:
    const sdfg::math::blas::GEMMNode& gemm_node_;

    void emit_dlpack_tensor_wrapper(
        sdfg::codegen::PrettyPrinter& stream,
        const std::string& var_name,
        const std::string& data_ptr,
        const std::string& rows,
        const std::string& cols
    );

public:
    GEMMNodeDispatcher_QANT(
        sdfg::codegen::LanguageExtension& language_extension,
        const sdfg::Function& function,
        const sdfg::data_flow::DataFlowGraph& data_flow_graph,
        const sdfg::math::blas::GEMMNode& node
    );

    void dispatch_code(
        sdfg::codegen::PrettyPrinter& stream,
        sdfg::codegen::PrettyPrinter& globals_stream,
        sdfg::codegen::CodeSnippetFactory& library_snippet_factory
    ) override;
};

} // namespace blas
} // namespace qant
} // namespace docc
