#include "docc/qant/qant_offloaded_visitor.h"

namespace docc::qant {

bool QantOffloadedVisitor::visit(sdfg::structured_control_flow::Block& node) {
    auto& dflow = node.dataflow();
    for (auto* lib_node : dflow.library_nodes()) {
        if (lib_node->implementation_type() == ImplementationType_QANT) {
            offloaded_.push_back(OffloadedNodeInfo{
                .node = lib_node,
                .enclosing_loop = loop_stack_.empty() ? nullptr : loop_stack_.back(),
            });
        }
    }
    return true;
}

bool QantOffloadedVisitor::handleStructuredLoop(sdfg::structured_control_flow::StructuredLoop& loop) {
    loop_stack_.push_back(&loop);
    // Delegate to the default which visits the loop body sequence.
    ActualStructuredSDFGVisitor::handleStructuredLoop(loop);
    loop_stack_.pop_back();
    return true;
}

bool QantOffloadedVisitor::visit(sdfg::structured_control_flow::While& node) {
    loop_stack_.push_back(&node);
    ActualStructuredSDFGVisitor::visit(node);
    loop_stack_.pop_back();
    return true;
}

} // namespace docc::qant
