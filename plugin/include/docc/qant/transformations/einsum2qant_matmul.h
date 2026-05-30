#pragma once

#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/einsum_node.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

class Einsum2QantMatmul : public Transformation {
private:
    math::tensor::EinsumNode& einsum_node_;
    const std::string target_tune_;

    bool check_matrix_indices(long long mat, const symbolic::Symbol& indvar1, const symbolic::Symbol& indvar2);
    bool is_qant_target() const;

public:
    Einsum2QantMatmul(math::tensor::EinsumNode& einsum_node, const std::string& target_tune);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static Einsum2QantMatmul from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace sdfg
