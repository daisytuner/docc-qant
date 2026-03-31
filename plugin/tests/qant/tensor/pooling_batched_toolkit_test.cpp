/**
 * @file pooling_batched_toolkit_test.cpp
 *
 * Minimal reproducer for batched pooling input bug in the QANT native
 * computing toolkit.  Calls maxpool2d_fprop and avgpool2d_fprop directly
 * with n_batches > 1.  Both API docs state that the features tensor is
 * 4D with shape (n_batches, n_channels, height, width), so batched input
 * should be supported.
 *
 * Expected: the toolkit processes all batches and returns correct results.
 * Actual:   "does not support batched input" error at runtime.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <stdfloat>
#include <vector>

#include <dlpack/dlpack.h>
#include <qant_native_computing_toolkit.h>

namespace {

/// Helper: build a row-major 4D DLPack tensor wrapper (does not own data).
DLManagedTensorVersioned make_tensor_4d(
    void* data, int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3, int64_t* shape_buf, int64_t* stride_buf
) {
    shape_buf[0] = dim0;
    shape_buf[1] = dim1;
    shape_buf[2] = dim2;
    shape_buf[3] = dim3;

    stride_buf[0] = dim1 * dim2 * dim3;
    stride_buf[1] = dim2 * dim3;
    stride_buf[2] = dim3;
    stride_buf[3] = 1;

    DLManagedTensorVersioned t{};
    t.version.major = DLPACK_MAJOR_VERSION;
    t.version.minor = DLPACK_MINOR_VERSION;
    t.manager_ctx = nullptr;
    t.deleter = nullptr;
    t.flags = 0;
    t.dl_tensor.data = data;
    t.dl_tensor.device.device_type = kDLCPU;
    t.dl_tensor.device.device_id = 0;
    t.dl_tensor.ndim = 4;
    t.dl_tensor.dtype.code = kDLBfloat;
    t.dl_tensor.dtype.bits = 16;
    t.dl_tensor.dtype.lanes = 1;
    t.dl_tensor.shape = shape_buf;
    t.dl_tensor.strides = stride_buf;
    t.dl_tensor.byte_offset = 0;
    return t;
}

} // namespace


// ---------------------------------------------------------------------------
// maxpool2d_fprop with n_batches = 2
// ---------------------------------------------------------------------------
TEST(ToolkitPoolingBatchedBug, MaxPool2D_Batched) {
    const int N = 1, C = 1, H = 4, W = 4;
    const size_t kH = 2, kW = 2, stride = 2, padding = 0;
    const int H_out = (H + 2 * static_cast<int>(padding) - static_cast<int>(kH)) / static_cast<int>(stride) + 1;
    const int W_out = (W + 2 * static_cast<int>(padding) - static_cast<int>(kW)) / static_cast<int>(stride) + 1;

    const int total_in = N * C * H * W;

    std::vector<__bf16> X(total_in);
    for (int i = 0; i < total_in; ++i) {
        X[i] = static_cast<__bf16>((i + 1) * 0.1f);
    }

    int64_t shape[4], strides_buf[4];
    auto tensor = make_tensor_4d(X.data(), N, C, H, W, shape, strides_buf);

    const uint32_t npu_id = 0;

    DLManagedTensorVersioned* result =
        qant_native_computing_toolkit::maxpool2d_fprop(npu_id, &tensor, kH, kW, padding, stride);

    ASSERT_NE(result, nullptr) << "maxpool2d_fprop returned nullptr for batched input (n_batches=2)";

    // Verify shape
    ASSERT_EQ(result->dl_tensor.ndim, 4);
    EXPECT_EQ(result->dl_tensor.shape[0], N);
    EXPECT_EQ(result->dl_tensor.shape[1], C);
    EXPECT_EQ(result->dl_tensor.shape[2], H_out);
    EXPECT_EQ(result->dl_tensor.shape[3], W_out);

    auto* out = reinterpret_cast<__bf16*>(result->dl_tensor.data);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    float ref_max = -1e30f;
                    for (size_t kh_i = 0; kh_i < kH; ++kh_i) {
                        for (size_t kw_i = 0; kw_i < kW; ++kw_i) {
                            int ih = oh * static_cast<int>(stride) + static_cast<int>(kh_i);
                            int iw = ow * static_cast<int>(stride) + static_cast<int>(kw_i);
                            float val = static_cast<float>(X[((n * C + c) * H + ih) * W + iw]);
                            if (val > ref_max) ref_max = val;
                        }
                    }
                    int idx = ((n * C + c) * H_out + oh) * W_out + ow;
                    EXPECT_NEAR(static_cast<float>(out[idx]), ref_max, 5e-2f)
                        << "maxpool mismatch at batch=" << n << " idx=" << idx;
                }
            }
        }
    }

    if (result->deleter) result->deleter(result);
}


// ---------------------------------------------------------------------------
// avgpool2d_fprop with n_batches = 2
// ---------------------------------------------------------------------------
TEST(ToolkitPoolingBatchedBug, AvgPool2D_Batched) {
    const int N = 1, C = 1, H = 4, W = 4;
    const size_t kH = 2, kW = 2, stride = 2, padding = 0;
    const bool count_include_pad = false;
    const int H_out = (H + 2 * static_cast<int>(padding) - static_cast<int>(kH)) / static_cast<int>(stride) + 1;
    const int W_out = (W + 2 * static_cast<int>(padding) - static_cast<int>(kW)) / static_cast<int>(stride) + 1;

    const int total_in = N * C * H * W;

    std::vector<__bf16> X(total_in);
    for (int i = 0; i < total_in; ++i) {
        X[i] = static_cast<__bf16>((i + 1) * 0.1f);
    }

    int64_t shape[4], strides_buf[4];
    auto tensor = make_tensor_4d(X.data(), N, C, H, W, shape, strides_buf);

    const uint32_t npu_id = 0;

    DLManagedTensorVersioned* result =
        qant_native_computing_toolkit::avgpool2d_fprop(npu_id, &tensor, kH, kW, padding, stride, count_include_pad);

    ASSERT_NE(result, nullptr) << "avgpool2d_fprop returned nullptr for batched input (n_batches=2)";

    ASSERT_EQ(result->dl_tensor.ndim, 4);
    EXPECT_EQ(result->dl_tensor.shape[0], N);
    EXPECT_EQ(result->dl_tensor.shape[1], C);
    EXPECT_EQ(result->dl_tensor.shape[2], H_out);
    EXPECT_EQ(result->dl_tensor.shape[3], W_out);

    auto* out = reinterpret_cast<__bf16*>(result->dl_tensor.data);
    const float kernel_area = static_cast<float>(kH * kW);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    float sum = 0.0f;
                    for (size_t kh_i = 0; kh_i < kH; ++kh_i) {
                        for (size_t kw_i = 0; kw_i < kW; ++kw_i) {
                            int ih = oh * static_cast<int>(stride) + static_cast<int>(kh_i);
                            int iw = ow * static_cast<int>(stride) + static_cast<int>(kw_i);
                            sum += static_cast<float>(X[((n * C + c) * H + ih) * W + iw]);
                        }
                    }
                    float ref_avg = sum / kernel_area;
                    int idx = ((n * C + c) * H_out + oh) * W_out + ow;
                    EXPECT_NEAR(static_cast<float>(out[idx]), ref_avg, 5e-2f)
                        << "avgpool mismatch at batch=" << n << " idx=" << idx;
                }
            }
        }
    }

    if (result->deleter) result->deleter(result);
}
