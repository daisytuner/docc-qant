#pragma once

#include <vector>

#include "sdfg/data_flow/library_node.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

#include "docc/qant/qant.h"

namespace docc::qant {

/**
 * @brief Information about a single Qant-offloaded library node.
 */
struct OffloadedNodeInfo {
    const sdfg::data_flow::LibraryNode* node;
    /// Innermost enclosing loop scope (For, Map, or While), nullptr if none.
    const sdfg::structured_control_flow::ControlFlowNode* enclosing_loop;
};

/**
 * @brief Visitor that collects all library nodes with Qant implementation type.
 *
 * For each node it also records the innermost enclosing loop (if present).
 * We don't currently expect offloaded nodes inside loops, but tracking this
 * lets us detect that case and flag it for future revision.
 */
class QantOffloadedVisitor : public sdfg::visitor::ActualStructuredSDFGVisitor {
    /// Stack of loop scopes (For, Map, While), innermost last.
    std::vector<const sdfg::structured_control_flow::ControlFlowNode*> loop_stack_;

    /// Collected results.
    std::vector<OffloadedNodeInfo> offloaded_;

public:
    bool visit(sdfg::structured_control_flow::Block& node) override;

    bool handleStructuredLoop(sdfg::structured_control_flow::StructuredLoop& loop) override;

    bool visit(sdfg::structured_control_flow::While& node) override;

    /// Access the collected offloaded nodes after dispatch().
    const std::vector<OffloadedNodeInfo>& offloaded() const { return offloaded_; }
};

} // namespace docc::qant
