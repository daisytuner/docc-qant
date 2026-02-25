#pragma once

#include <sdfg/builder/sdfg_builder.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/data_flow/data_flow_graph.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/passes/pass.h>
#include <sdfg/structured_control_flow/block.h>
#include <sdfg/visitor/structured_sdfg_visitor.h>

#include "docc/qant/dataflow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/codegen/code_snippet_factory.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/types/type.h"

namespace docc::qant {

class ReduceQuantizationVisitor : public sdfg::visitor::ActualStructuredSDFGVisitor {
    friend class ReduceQuantizationPass;

    struct EdgeQuantizationInfo {
        const sdfg::data_flow::Memlet& memlet;
        std::optional<sdfg::types::PrimitiveType> reduced_src_type;
        std::optional<sdfg::types::PrimitiveType> reduced_dst_type;
    };

    struct ContainerQuantizationInfo {};

protected:
    sdfg::builder::StructuredSDFGBuilder& builder_;
    sdfg::analysis::AnalysisManager& analysis_manager_;

    std::unordered_map<size_t, EdgeQuantizationInfo> edge_queue_;
    std::unordered_map<std::string, ContainerQuantizationInfo> container_queue_;

public:
    ReduceQuantizationVisitor(
        sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager
    );

    bool visit(sdfg::structured_control_flow::Block& node) override;

    bool filter(sdfg::data_flow::LibraryNode& node);

    void check_in_edge_for_modification(
        const sdfg::data_flow::DataFlowGraph& dflow,
        const sdfg::data_flow::LibraryNode& node,
        sdfg::types::PrimitiveType new_op_type,
        const sdfg::data_flow::Memlet* memlet
    );

    void check_out_edges_for_modification(
        const sdfg::data_flow::DataFlowGraph& dflow,
        const sdfg::data_flow::LibraryNode& node,
        sdfg::types::PrimitiveType new_op_type,
        std::vector<const sdfg::data_flow::Memlet*> memlets
    );

    bool try_reduce(
        sdfg::data_flow::LibraryNode& node,
        sdfg::data_flow::DataFlowGraph& dflow,
        sdfg::builder::StructuredSDFGBuilder& builder
    );
};

class ReduceQuantizationPass : public sdfg::passes::Pass {
public:
    ReduceQuantizationPass();

    std::string name() override;

    bool run_pass(sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager)
        override;
};

struct CodegenOutput {
    sdfg::codegen::PrettyPrinter& main;
    sdfg::codegen::PrettyPrinter& globals;
    sdfg::codegen::CodeSnippetFactory& library_snippet_factory;
};

class DataIterator {
protected:
    const std::string var_name_;

public:
    DataIterator(const std::string& var_name);
    virtual ~DataIterator() = default;

    virtual sdfg::structured_control_flow::Block& build_sdfg_iteration(sdfg::builder::StructuredSDFGBuilder& builder
    ) = 0;
};

class QuantConversion {
public:
    QuantConversion();


    static void emit_conversion(
        CodegenOutput& output,
        sdfg::codegen::LanguageExtension& language_extension,
        const std::string& input_var,
        const sdfg::types::IType& input_type,
        const std::string& output_var,
        const sdfg::types::IType& output_type,
        sdfg::symbolic::Expression it_start,
        sdfg::symbolic::Expression it_end
    );
};

} // namespace docc::qant
