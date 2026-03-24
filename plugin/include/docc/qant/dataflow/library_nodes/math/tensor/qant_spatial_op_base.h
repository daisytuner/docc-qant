#pragma once

#include <cassert>
#include <cstddef>
#include <vector>

#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace math {
namespace tensor {

/**
 * @brief CRTP mixin providing shared spatial‐dimension utilities for Qant nodes.
 *
 * Both QantConvNode and QantPoolingNode operate on tensors with layout
 * [N, C, D1, …, Dn] and share identical kernel_shape / strides / pads /
 * dilations semantics.  This mixin extracts the common helpers so they
 * are written — and tested — in one place.
 *
 * ## Requirements on `Derived`
 *
 * `Derived` must expose the following **public** const accessors (both
 * ConvNode and PoolingNode already do):
 *
 * | Accessor          | Return type                                      |
 * |-------------------|--------------------------------------------------|
 * | `shape()`         | `const std::vector<symbolic::Expression>&`       |
 * | `kernel_shape()`  | `const std::vector<symbolic::Expression>&`       |
 * | `strides()`       | `const std::vector<symbolic::Expression>&`       |
 * | `pads()`          | `const std::vector<symbolic::Expression>&`       |
 * | `dilations()`     | `const std::vector<symbolic::Expression>&`       |
 */
template<typename Derived>
class QantSpatialOpBase {
protected:
    /** Helper: down-cast to the concrete node type. */
    const Derived& derived() const { return *static_cast<const Derived*>(this); }

public:
    /**
     * @brief Number of spatial dimensions.
     *
     * For a tensor with shape [N, C, D1, …, Dn] this returns n.
     */
    size_t num_spatial_dims() const {
        auto& s = derived().shape();
        assert(s.size() >= 2);
        return s.size() - 2;
    }

    /**
     * @brief Compute the output size for spatial dimension @p i.
     *
     * Uses the standard convolution / pooling output-size formula:
     * @code
     *   floor((D_i + pad_begin_i + pad_end_i
     *          - dilation_i * (k_i - 1) - 1) / stride_i) + 1
     * @endcode
     *
     * @param i  Spatial dimension index (0-based, relative to the spatial axes).
     */
    symbolic::Expression output_spatial_dim(size_t i) const {
        size_t n_spatial = num_spatial_dims();
        assert(i < n_spatial);

        auto& s = derived().shape();
        auto& ks = derived().kernel_shape();
        auto& st = derived().strides();
        auto& pa = derived().pads();
        auto& di = derived().dilations();

        auto d_in = s[2 + i];
        auto k = ks[i];

        symbolic::Expression stride = st.empty() ? symbolic::Expression(symbolic::one()) : st[i];
        symbolic::Expression dilation = di.empty() ? symbolic::Expression(symbolic::one()) : di[i];

        // pads layout: [begin_d0, begin_d1, …, end_d0, end_d1, …]
        symbolic::Expression pad_begin = pa.empty() ? symbolic::Expression(symbolic::zero()) : pa[i];
        symbolic::Expression pad_end = pa.empty() ? symbolic::Expression(symbolic::zero()) : pa[n_spatial + i];

        // numerator = D_i + pad_begin + pad_end - dilation * (k - 1) - 1
        auto numerator = symbolic::
            sub(symbolic::add(symbolic::add(d_in, pad_begin), pad_end),
                symbolic::add(symbolic::mul(dilation, symbolic::sub(k, symbolic::one())), symbolic::one()));

        return symbolic::add(symbolic::div(numerator, stride), symbolic::one());
    }

    /**
     * @brief Total number of output spatial elements: prod(output_spatial_dim(i)).
     *
     * Does **not** include the batch (N) or channel (C / C_out) dimensions —
     * those are node-specific.
     */
    symbolic::Expression output_spatial_volume() const {
        size_t n_spatial = num_spatial_dims();
        symbolic::Expression result = symbolic::Expression(symbolic::one());
        for (size_t i = 0; i < n_spatial; ++i) {
            result = symbolic::mul(result, output_spatial_dim(i));
        }
        return result;
    }

    /**
     * @brief Total number of kernel / window elements: prod(kernel_shape[i]).
     *
     * For convolutions this is the per-channel kernel volume;
     * for pooling this is the window volume.
     */
    symbolic::Expression kernel_volume() const {
        auto& ks = derived().kernel_shape();
        return SymEngine::mul(ks);
    }
};

} // namespace tensor
} // namespace math
} // namespace sdfg
