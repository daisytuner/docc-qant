#include <gtest/gtest.h>

#include <dlfcn.h>
#include <vector>

#include <sdfg/passes/dataflow/tensor_to_pointer_conversion.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/conv_node.h"
#include "sdfg/passes/pipeline.h"
#include "sdfg_debug_dump.h"
#include "so_compile.h"

#include "docc/qant/passes/remapping_pass.h"
#include "docc/qant/plugin.h"
#include "docc/qant/qant.h"

using namespace sdfg;


TEST(ConvTest, Conv2D_QANT_Simple) {
    // Test simple 2D convolution via QANT: X[1, 1, 5, 5] * W[1, 1, 3, 3] = Y[1, 1, 3, 3]
    // No padding, stride=1, dilation=1
    builder::StructuredSDFGBuilder builder("conv2d_qant_simple", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::BFloat);
    types::Pointer desc_ptr(desc);

    builder.add_container("x", desc_ptr, true);
    builder.add_container("w", desc_ptr, true);
    builder.add_container("y", desc_ptr, true);

    auto& block = builder.add_block(sdfg.root());

    auto& x_node = builder.add_access(block, "x");
    auto& w_node = builder.add_access(block, "w");
    auto& y_node = builder.add_access(block, "y");

    const int N = 1, C_in = 1, H = 5, W = 5;
    const int C_out = 1, kH = 3, kW = 3;
    const int stride_val = 1, padding_val = 0, dilation_val = 1;
    const int H_out = (H + 2 * padding_val - dilation_val * (kH - 1) - 1) / stride_val + 1;
    const int W_out = (W + 2 * padding_val - dilation_val * (kW - 1) - 1) / stride_val + 1;

    std::vector<symbolic::Expression> input_shape = {
        symbolic::integer(N), symbolic::integer(C_in), symbolic::integer(H), symbolic::integer(W)
    };
    std::vector<symbolic::Expression> kernel_shape = {symbolic::integer(kH), symbolic::integer(kW)};
    std::vector<symbolic::Expression> strides_vec = {symbolic::integer(stride_val), symbolic::integer(stride_val)};
    std::vector<symbolic::Expression> pads_vec = {
        symbolic::integer(padding_val),
        symbolic::integer(padding_val),
        symbolic::integer(padding_val),
        symbolic::integer(padding_val)
    };
    std::vector<symbolic::Expression> dilations_vec = {
        symbolic::integer(dilation_val), symbolic::integer(dilation_val)
    };

    types::Tensor input_tensor_x(desc.primitive_type(), input_shape);
    symbolic::MultiExpression weight_shape = {
        symbolic::integer(C_out), symbolic::integer(C_in), symbolic::integer(kH), symbolic::integer(kW)
    };
    types::Tensor input_tensor_w(desc.primitive_type(), weight_shape);
    symbolic::MultiExpression output_shape = {
        symbolic::integer(N), symbolic::integer(C_out), symbolic::integer(H_out), symbolic::integer(W_out)
    };
    types::Tensor output_tensor(desc.primitive_type(), output_shape);

    auto& conv_node = static_cast<math::tensor::ConvNode&>(builder.add_library_node<math::tensor::ConvNode>(
        block,
        DebugInfo(),
        input_shape,
        kernel_shape,
        strides_vec,
        pads_vec,
        dilations_vec,
        symbolic::integer(C_out),
        symbolic::integer(1)
    ));

    builder.add_computational_memlet(block, x_node, conv_node, "X", {}, input_tensor_x, block.debug_info());
    builder.add_computational_memlet(block, w_node, conv_node, "W", {}, input_tensor_w, block.debug_info());
    builder.add_computational_memlet(block, y_node, conv_node, "Y", {}, output_tensor, block.debug_info());

    sdfg.validate();

    analysis::AnalysisManager analysis_manager(sdfg);

    // Run remapping pass: ConvNode → QantConvNode
    sdfg::passes::Pipeline remapping("QantRemapping");
    remapping.register_pass<sdfg::passes::QantRemappingPass>();
    remapping.run(builder, analysis_manager);

    sdfg.validate();

    auto library_nodes = block.dataflow().library_nodes();
    EXPECT_EQ(library_nodes.size(), 1);
    auto* new_node = dynamic_cast<sdfg::math::tensor::ConvNode*>(*library_nodes.begin());
    EXPECT_TRUE(new_node);
    EXPECT_EQ(new_node->implementation_type(), docc::qant::ImplementationType_QANT.value());

    sdfg::passes::TensorToPointerConversionPass tensor_to_pointer_conversion_pass;
    tensor_to_pointer_conversion_pass.run(builder, analysis_manager);

    sdfg.validate();

    docc::qant::schedule(sdfg, "qant");

    sdfg.validate();

    std::string lib_path = docc::qant::so_compile(sdfg);

    void* h = dlopen(lib_path.c_str(), RTLD_LAZY);
    ASSERT_NE(h, nullptr) << dlerror();

    using Fn = void (*)(__bf16*, __bf16*, __bf16*);
    auto fn = reinterpret_cast<Fn>(dlsym(h, sdfg.name().c_str()));
    ASSERT_NE(fn, nullptr) << dlerror();

    const int total_x = N * C_in * H * W;
    const int total_w = C_out * C_in * kH * kW;
    const int total_y = N * C_out * H_out * W_out;

    std::vector<__bf16> X_data(total_x), W_data(total_w), Y_result(total_y, (__bf16) 0.0f);
    std::vector<float> Y_ref(total_y, 0.0f);

    for (int i = 0; i < total_x; ++i) X_data[i] = (__bf16) (static_cast<float>(i + 1) * 0.1f);
    for (int i = 0; i < total_w; ++i) W_data[i] = (__bf16) (static_cast<float>(i + 1) * 0.1f);

    // Reference conv2d (no padding, stride=1, dilation=1)
    for (int n = 0; n < N; ++n) {
        for (int co = 0; co < C_out; ++co) {
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    float sum = 0.0f;
                    for (int ci = 0; ci < C_in; ++ci) {
                        for (int kh_i = 0; kh_i < kH; ++kh_i) {
                            for (int kw_i = 0; kw_i < kW; ++kw_i) {
                                int ih = oh + kh_i;
                                int iw = ow + kw_i;
                                float x_val = static_cast<float>(X_data[((n * C_in + ci) * H + ih) * W + iw]);
                                float w_val = static_cast<float>(W_data[((co * C_in + ci) * kH + kh_i) * kW + kw_i]);
                                sum += x_val * w_val;
                            }
                        }
                    }
                    Y_ref[((n * C_out + co) * H_out + oh) * W_out + ow] = sum;
                }
            }
        }
    }

    fn(X_data.data(), W_data.data(), Y_result.data());

    for (int i = 0; i < total_y; ++i) {
        EXPECT_NEAR(static_cast<float>(Y_result[i]), Y_ref[i], 5e-2f) << "Mismatch at index " << i;
    }

    dlclose(h);
}

TEST(ConvTest, Conv2D_QANT_WithPadding) {
    // Test 2D convolution with padding via QANT: X[1, 2, 4, 4] * W[3, 2, 3, 3] = Y[1, 3, 4, 4]
    // padding=1, stride=1, dilation=1
    builder::StructuredSDFGBuilder builder("conv2d_qant_padded", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::BFloat);
    types::Pointer desc_ptr(desc);

    builder.add_container("x", desc_ptr, true);
    builder.add_container("w", desc_ptr, true);
    builder.add_container("y", desc_ptr, true);

    auto& block = builder.add_block(sdfg.root());

    auto& x_node = builder.add_access(block, "x");
    auto& w_node = builder.add_access(block, "w");
    auto& y_node = builder.add_access(block, "y");

    const int N = 1, C_in = 2, H = 4, W = 4;
    const int C_out = 3, kH = 3, kW = 3;
    const int stride_val = 1, padding_val = 1, dilation_val = 1;
    const int H_out = (H + 2 * padding_val - dilation_val * (kH - 1) - 1) / stride_val + 1;
    const int W_out = (W + 2 * padding_val - dilation_val * (kW - 1) - 1) / stride_val + 1;

    std::vector<symbolic::Expression> input_shape = {
        symbolic::integer(N), symbolic::integer(C_in), symbolic::integer(H), symbolic::integer(W)
    };
    std::vector<symbolic::Expression> kernel_shape = {symbolic::integer(kH), symbolic::integer(kW)};
    std::vector<symbolic::Expression> strides_vec = {symbolic::integer(stride_val), symbolic::integer(stride_val)};
    std::vector<symbolic::Expression> pads_vec = {
        symbolic::integer(padding_val),
        symbolic::integer(padding_val),
        symbolic::integer(padding_val),
        symbolic::integer(padding_val)
    };
    std::vector<symbolic::Expression> dilations_vec = {
        symbolic::integer(dilation_val), symbolic::integer(dilation_val)
    };

    types::Tensor input_tensor_x(desc.primitive_type(), input_shape);
    symbolic::MultiExpression weight_shape = {
        symbolic::integer(C_out), symbolic::integer(C_in), symbolic::integer(kH), symbolic::integer(kW)
    };
    types::Tensor input_tensor_w(desc.primitive_type(), weight_shape);
    symbolic::MultiExpression output_shape = {
        symbolic::integer(N), symbolic::integer(C_out), symbolic::integer(H_out), symbolic::integer(W_out)
    };
    types::Tensor output_tensor(desc.primitive_type(), output_shape);

    auto& conv_node = static_cast<math::tensor::ConvNode&>(builder.add_library_node<math::tensor::ConvNode>(
        block,
        DebugInfo(),
        input_shape,
        kernel_shape,
        strides_vec,
        pads_vec,
        dilations_vec,
        symbolic::integer(C_out),
        symbolic::integer(1)
    ));

    builder.add_computational_memlet(block, x_node, conv_node, "X", {}, input_tensor_x, block.debug_info());
    builder.add_computational_memlet(block, w_node, conv_node, "W", {}, input_tensor_w, block.debug_info());
    builder.add_computational_memlet(block, y_node, conv_node, "Y", {}, output_tensor, block.debug_info());

    sdfg.validate();

    analysis::AnalysisManager analysis_manager(sdfg);

    sdfg::passes::Pipeline remapping("QantRemapping");
    remapping.register_pass<sdfg::passes::QantRemappingPass>();
    remapping.run(builder, analysis_manager);

    sdfg.validate();

    auto library_nodes = block.dataflow().library_nodes();
    EXPECT_EQ(library_nodes.size(), 1);
    auto* new_node = dynamic_cast<sdfg::math::tensor::ConvNode*>(*library_nodes.begin());
    EXPECT_TRUE(new_node);
    EXPECT_EQ(new_node->implementation_type(), docc::qant::ImplementationType_QANT.value());

    sdfg::passes::TensorToPointerConversionPass tensor_to_pointer_conversion_pass;
    tensor_to_pointer_conversion_pass.run(builder, analysis_manager);

    sdfg.validate();

    docc::qant::schedule(sdfg, "qant");

    sdfg.validate();

    std::string lib_path = docc::qant::so_compile(sdfg);

    void* h = dlopen(lib_path.c_str(), RTLD_LAZY);
    ASSERT_NE(h, nullptr) << dlerror();

    using Fn = void (*)(__bf16*, __bf16*, __bf16*);
    auto fn = reinterpret_cast<Fn>(dlsym(h, sdfg.name().c_str()));
    ASSERT_NE(fn, nullptr) << dlerror();

    const int total_x = N * C_in * H * W;
    const int total_w = C_out * C_in * kH * kW;
    const int total_y = N * C_out * H_out * W_out;

    std::vector<__bf16> X_data(total_x), W_data(total_w), Y_result(total_y, (__bf16) 0.0f);
    std::vector<float> Y_ref(total_y, 0.0f);

    for (int i = 0; i < total_x; ++i) X_data[i] = (__bf16) (static_cast<float>(i % 7) * 0.3f);
    for (int i = 0; i < total_w; ++i) W_data[i] = (__bf16) (static_cast<float>(i % 5) * 0.2f);

    // Reference conv2d with padding
    for (int n = 0; n < N; ++n) {
        for (int co = 0; co < C_out; ++co) {
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    float sum = 0.0f;
                    for (int ci = 0; ci < C_in; ++ci) {
                        for (int kh_i = 0; kh_i < kH; ++kh_i) {
                            for (int kw_i = 0; kw_i < kW; ++kw_i) {
                                int ih = oh * stride_val - padding_val + kh_i * dilation_val;
                                int iw = ow * stride_val - padding_val + kw_i * dilation_val;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    float x_val = static_cast<float>(X_data[((n * C_in + ci) * H + ih) * W + iw]);
                                    float w_val = static_cast<float>(W_data[((co * C_in + ci) * kH + kh_i) * kW + kw_i]
                                    );
                                    sum += x_val * w_val;
                                }
                            }
                        }
                    }
                    Y_ref[((n * C_out + co) * H_out + oh) * W_out + ow] = sum;
                }
            }
        }
    }

    fn(X_data.data(), W_data.data(), Y_result.data());

    for (int i = 0; i < total_y; ++i) {
        EXPECT_NEAR(static_cast<float>(Y_result[i]), Y_ref[i], 5e-2f) << "Mismatch at index " << i;
    }

    dlclose(h);
}
