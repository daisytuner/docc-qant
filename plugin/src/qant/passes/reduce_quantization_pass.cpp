#include "docc/qant/passes/reduce_quantization_pass.h"

#include "docc/qant/dataflow/library_nodes/math/tensor/matmul_node.h"
#include "docc/qant/qant.h"
#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/codegen/language_extensions/cpp_language_extension.h"
#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"

namespace docc::qant {

using namespace sdfg;

ReduceQuantizationVisitor::ReduceQuantizationVisitor(
    sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager
)
    : builder_(builder), analysis_manager_(analysis_manager) {}

bool ReduceQuantizationVisitor::visit(sdfg::structured_control_flow::Block& node) {
    auto& dflow = node.dataflow();
    for (auto* lib_node : dflow.library_nodes()) {
        if (filter(*lib_node)) {
            try_reduce(*lib_node, dflow, builder_);
        }
    }
    return true;
}

bool ReduceQuantizationVisitor::try_reduce(
    sdfg::data_flow::LibraryNode& node,
    sdfg::data_flow::DataFlowGraph& dflow,
    sdfg::builder::StructuredSDFGBuilder& builder
) {
    auto& code = node.code();
    if (code == sdfg::math::tensor::LibraryNodeType_QantMatMul) {
        auto& matmul_node = dynamic_cast<sdfg::math::tensor::QantMatMulNode&>(node);
        if (matmul_node.quantization() == sdfg::types::Float) {
            matmul_node.set_quantization(sdfg::types::BFloat);
        }

        // queue edges!
    }

    return false;
}

bool ReduceQuantizationVisitor::filter(sdfg::data_flow::LibraryNode& node) {
    auto type = node.implementation_type() == docc::qant::ImplementationType_QANT;


    return type;
}

ReduceQuantizationPass::ReduceQuantizationPass() : Pass() {}

std::string ReduceQuantizationPass::name() { return "ReduceQuantizationPass"; }

bool ReduceQuantizationPass::
    run_pass(sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager) {
    ReduceQuantizationVisitor visitor(builder, analysis_manager);
    visitor.dispatch(builder.subject().root());

    auto& edges = visitor.edge_queue_;

    // edge reduction visitor here. For now, let dispatchers do the work

    return edges.size() > 0;
}

void QuantConversion::emit_conversion(
    CodegenOutput& output,
    sdfg::codegen::LanguageExtension& language_extension,
    const std::string& input_var,
    const sdfg::types::IType& input_type,
    const std::string& output_var,
    const sdfg::types::IType& output_type,
    symbolic::Expression it_start,
    symbolic::Expression it_end
) {
    builder::StructuredSDFGBuilder builder("hidden", FunctionType_CPU);

    // output.main << "{" << std::endl;
    // output.main.changeIndent(+4);
    auto it_name = "it";
    types::Scalar it_type(types::PrimitiveType::UInt64);
    auto c_type = language_extension.declaration("", it_type);
    // output.main << c_type << " " << it_name << ";" << std::endl;

    output.main << "for (" << c_type << " " << it_name << " = " << language_extension.expression(it_start) << "; "
                << it_name << " < " << language_extension.expression(it_end) << "; ++" << it_name << ") {" << std::endl;
    output.main.changeIndent(+4);

    auto target_type = language_extension.primitive_type(output_type.primitive_type());

    output.main << output_var << "[" << it_name << "] = static_cast<" << target_type << ">(" << input_var << "["
                << it_name << "]);" << std::endl;

    // auto& root_scope = builder.subject().root();
    // builder.add_container(it_name, types::Scalar(input_type.primitive_type()), false);
    // builder.add_container(input_var, input_type, false);
    // builder.add_container(output_var, output_type, false);
    //
    // auto indvar_sym = symbolic::symbol(it_name);
    // auto& it_scope = builder.add_map(builder.subject().root(), indvar_sym, symbolic::Lt(indvar_sym, it_end),
    // it_start, symbolic::add(indvar_sym, symbolic::integer(1)),
    // structured_control_flow::ScheduleType_Sequential::create()); auto& it_block = builder.add_block(it_scope.root());
    // auto& src_node = builder.add_access(it_block, input_var);
    // auto& dst_node = builder.add_access(it_block, output_var);
    // auto& copy_op = builder.add_tasklet(it_block, data_flow::assign, "_ret", {"_in"});
    // builder.add_computational_memlet(it_block, src_node, copy_op, "_in", {indvar_sym}, input_type);
    // builder.add_computational_memlet(it_block, copy_op, "_ret", dst_node, {indvar_sym}, output_type);
    //
    //
    // analysis::AnalysisManager analysis_manager(builder.subject());
    // codegen::CPPLanguageExtension cpp_language_extension(builder.subject());
    // auto instrumentation_plan = codegen::InstrumentationPlan::none(builder.subject());
    // auto arg_capture_plan = codegen::ArgCapturePlan::none(builder.subject());
    // auto dispatcher = codegen::create_dispatcher(
    //     cpp_language_extension,
    //     builder.subject(),
    //     analysis_manager,
    //     root_scope,
    //     *instrumentation_plan,
    //     *arg_capture_plan
    // );
    // dispatcher->dispatch(output.main, output.globals, output.library_snippet_factory);


    output.main.changeIndent(-4);
    output.main << "}" << std::endl;
}

} // namespace docc::qant
