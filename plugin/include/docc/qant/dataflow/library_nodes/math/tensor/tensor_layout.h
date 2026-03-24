#pragma once
#include <nlohmann/json_fwd.hpp>

#include "sdfg/symbolic/symbolic.h"

namespace sdfg::math::tensor {

class TensorLayout {
private:
    /**
     * Shape of input tensor [..., M, K]
     */
    symbolic::MultiExpression shape_;
    /**
     * Strides for tensor (defaults to row-major contiguous)
     */
    symbolic::MultiExpression strides_;
    /**
     * Offset into tensor in elements (defaults to 0)
     */
    symbolic::Expression offset_;

public:
    TensorLayout(
        const symbolic::MultiExpression& shape,
        const symbolic::MultiExpression& strides = {},
        const symbolic::Expression offset = symbolic::integer(0)
    );

    const symbolic::MultiExpression& shape() const { return shape_; }
    const symbolic::MultiExpression& strides() const { return strides_; }
    const symbolic::Expression& offset() const { return offset_; }

    void serialize_to_json(nlohmann::json& j) const;

    std::string toStr() const;

    symbolic::MultiExpression linear_strides() const;
    static symbolic::MultiExpression linear_strides(const symbolic::MultiExpression& shape);

    /**
     *
     * @param i 0 is innermost dim, 1 next level out etc.
     */
    symbolic::Expression get_dim_innermost(int i) const { return shape_.at(shape_.size() - 1 - i); }

    /**
     *
     * @param i 0 is innermost dim, 1 next level out etc.
     */
    symbolic::Expression get_stride_innermost(int i) const { return strides_.at(strides_.size() - 1 - i); }

    int dims() const { return shape_.size(); }

    static TensorLayout deserialize_from_json(const nlohmann::json& j);

    static bool has_linear_accesses_no_padding(symbolic::MultiExpression shape, symbolic::MultiExpression strides);

    bool has_linear_accesses_no_padding() const;

    bool has_transposed_strides_no_padding() const;
};

std::ostream& operator<<(std::ostream& stream, const TensorLayout& layout);


} // namespace sdfg::math::tensor
