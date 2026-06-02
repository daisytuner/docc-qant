#include <gtest/gtest.h>

#include <dlfcn.h>
#include <vector>

#include <sdfg/passes/dataflow/tensor_to_pointer_conversion.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/pooling_node.h"
#include "sdfg/passes/pipeline.h"
#include "sdfg_debug_dump.h"
#include "so_compile.h"

#include "docc/qant/passes/remapping_pass.h"
#include "docc/qant/plugin.h"
#include "docc/qant/qant.h"

using namespace sdfg;


TEST(PoolingTest, MaxPool2D_QANT_Simple) {
    // Test simple 2D max pooling via QANT: X[1, 1, 4, 4] -> Y[1, 1, 2, 2]
    // kernel=2x2, stride=2, no padding
    builder::StructuredSDFGBuilder builder("maxpool2d_qant_simple", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::BFloat);
    types::Pointer desc_ptr(desc);

    builder.add_container("x", desc_ptr, true);
    builder.add_container("y", desc_ptr, true);

    auto& block = builder.add_block(sdfg.root());

    auto& x_node = builder.add_access(block, "x");
    auto& y_node = builder.add_access(block, "y");

    const int N = 1, C = 1, H = 4, W = 4;
    const int kH = 2, kW = 2;
    const int stride_val = 2, padding_val = 0;
    const int H_out = (H + 2 * padding_val - kH) / stride_val + 1;
    const int W_out = (W + 2 * padding_val - kW) / stride_val + 1;

    std::vector<symbolic::Expression> input_shape = {
        symbolic::integer(N), symbolic::integer(C), symbolic::integer(H), symbolic::integer(W)
    };
    std::vector<symbolic::Expression> kernel_shape = {symbolic::integer(kH), symbolic::integer(kW)};
    std::vector<symbolic::Expression> strides_vec = {symbolic::integer(stride_val), symbolic::integer(stride_val)};
    std::vector<symbolic::Expression> pads_vec = {
        symbolic::integer(padding_val),
        symbolic::integer(padding_val),
        symbolic::integer(padding_val),
        symbolic::integer(padding_val)
    };
    std::vector<symbolic::Expression> dilations_vec = {symbolic::integer(1), symbolic::integer(1)};

    types::Tensor input_tensor_x(desc.primitive_type(), input_shape);
    symbolic::MultiExpression output_shape = {
        symbolic::integer(N), symbolic::integer(C), symbolic::integer(H_out), symbolic::integer(W_out)
    };
    types::Tensor output_tensor(desc.primitive_type(), output_shape);

    auto& pooling_node = static_cast<math::tensor::PoolingNode&>(builder.add_library_node<math::tensor::PoolingNode>(
        block, DebugInfo(), math::tensor::PoolingMode::Max, input_shape, kernel_shape, strides_vec, pads_vec, dilations_vec
    ));

    builder.add_computational_memlet(block, x_node, pooling_node, "X", {}, input_tensor_x, block.debug_info());
    builder.add_computational_memlet(block, y_node, pooling_node, "Y", {}, output_tensor, block.debug_info());

    sdfg.validate();

    analysis::AnalysisManager analysis_manager(sdfg);

    // Run remapping pass: PoolingNode → QantPoolingNode
    sdfg::passes::Pipeline remapping("QantRemapping");
    remapping.register_pass<sdfg::passes::QantRemappingPass>();
    remapping.run(builder, analysis_manager);

    sdfg.validate();

    auto library_nodes = block.dataflow().library_nodes();
    EXPECT_EQ(library_nodes.size(), 1);
    auto* new_node = dynamic_cast<sdfg::math::tensor::PoolingNode*>(*library_nodes.begin());
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

    using Fn = void (*)(__bf16*, __bf16*);
    auto fn = reinterpret_cast<Fn>(dlsym(h, sdfg.name().c_str()));
    ASSERT_NE(fn, nullptr) << dlerror();

    const int total_x = N * C * H * W;
    const int total_y = N * C * H_out * W_out;

    std::vector<__bf16> X_data(total_x), Y_result(total_y, (__bf16) 0.0f);
    std::vector<float> Y_ref(total_y, 0.0f);

    for (int i = 0; i < total_x; ++i) X_data[i] = (__bf16) (static_cast<float>(i + 1) * 0.1f);

    // Reference max pooling
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    float max_val = -1e30f;
                    for (int kh_i = 0; kh_i < kH; ++kh_i) {
                        for (int kw_i = 0; kw_i < kW; ++kw_i) {
                            int ih = oh * stride_val + kh_i;
                            int iw = ow * stride_val + kw_i;
                            float x_val = static_cast<float>(X_data[((n * C + c) * H + ih) * W + iw]);
                            if (x_val > max_val) max_val = x_val;
                        }
                    }
                    Y_ref[((n * C + c) * H_out + oh) * W_out + ow] = max_val;
                }
            }
        }
    }

    fn(X_data.data(), Y_result.data());

    for (int i = 0; i < total_y; ++i) {
        EXPECT_NEAR(static_cast<float>(Y_result[i]), Y_ref[i], 5e-2f) << "Mismatch at index " << i;
    }

    dlclose(h);
}


TEST(PoolingTest, AvgPool2D_QANT_Simple) {
    // Test simple 2D average pooling via QANT: X[1, 1, 4, 4] -> Y[1, 1, 2, 2]
    // kernel=2x2, stride=2, no padding, mode=Sum (maps to avgpool2d_fprop)
    builder::StructuredSDFGBuilder builder("avgpool2d_qant_simple", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::BFloat);
    types::Pointer desc_ptr(desc);

    builder.add_container("x", desc_ptr, true);
    builder.add_container("y", desc_ptr, true);

    auto& block = builder.add_block(sdfg.root());

    auto& x_node = builder.add_access(block, "x");
    auto& y_node = builder.add_access(block, "y");

    const int N = 1, C = 1, H = 4, W = 4;
    const int kH = 2, kW = 2;
    const int stride_val = 2, padding_val = 0;
    const int H_out = (H + 2 * padding_val - kH) / stride_val + 1;
    const int W_out = (W + 2 * padding_val - kW) / stride_val + 1;

    std::vector<symbolic::Expression> input_shape = {
        symbolic::integer(N), symbolic::integer(C), symbolic::integer(H), symbolic::integer(W)
    };
    std::vector<symbolic::Expression> kernel_shape = {symbolic::integer(kH), symbolic::integer(kW)};
    std::vector<symbolic::Expression> strides_vec = {symbolic::integer(stride_val), symbolic::integer(stride_val)};
    std::vector<symbolic::Expression> pads_vec = {
        symbolic::integer(padding_val),
        symbolic::integer(padding_val),
        symbolic::integer(padding_val),
        symbolic::integer(padding_val)
    };
    std::vector<symbolic::Expression> dilations_vec = {symbolic::integer(1), symbolic::integer(1)};

    types::Tensor input_tensor_x(desc.primitive_type(), input_shape);
    symbolic::MultiExpression output_shape = {
        symbolic::integer(N), symbolic::integer(C), symbolic::integer(H_out), symbolic::integer(W_out)
    };
    types::Tensor output_tensor(desc.primitive_type(), output_shape);

    auto& pooling_node = static_cast<math::tensor::PoolingNode&>(builder.add_library_node<math::tensor::PoolingNode>(
        block, DebugInfo(), math::tensor::PoolingMode::Sum, input_shape, kernel_shape, strides_vec, pads_vec, dilations_vec
    ));

    builder.add_computational_memlet(block, x_node, pooling_node, "X", {}, input_tensor_x, block.debug_info());
    builder.add_computational_memlet(block, pooling_node, "Y", y_node, {}, output_tensor, block.debug_info());

    sdfg.validate();

    analysis::AnalysisManager analysis_manager(sdfg);

    // Run remapping pass: PoolingNode → QantPoolingNode
    sdfg::passes::Pipeline remapping("QantRemapping");
    remapping.register_pass<sdfg::passes::QantRemappingPass>();
    remapping.run(builder, analysis_manager);

    sdfg.validate();

    auto library_nodes = block.dataflow().library_nodes();
    EXPECT_EQ(library_nodes.size(), 1);
    auto* new_node = dynamic_cast<sdfg::math::tensor::PoolingNode*>(*library_nodes.begin());
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

    using Fn = void (*)(__bf16*, __bf16*);
    auto fn = reinterpret_cast<Fn>(dlsym(h, sdfg.name().c_str()));
    ASSERT_NE(fn, nullptr) << dlerror();

    const int total_x = N * C * H * W;
    const int total_y = N * C * H_out * W_out;

    std::vector<__bf16> X_data(total_x), Y_result(total_y, (__bf16) 0.0f);
    std::vector<float> Y_ref(total_y, 0.0f);

    for (int i = 0; i < total_x; ++i) X_data[i] = (__bf16) (static_cast<float>(i + 1) * 0.1f);

    // Reference sum pooling (Sum mode computes raw sum, not average)
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    float sum = 0.0f;
                    for (int kh_i = 0; kh_i < kH; ++kh_i) {
                        for (int kw_i = 0; kw_i < kW; ++kw_i) {
                            int ih = oh * stride_val + kh_i;
                            int iw = ow * stride_val + kw_i;
                            sum += static_cast<float>(X_data[((n * C + c) * H + ih) * W + iw]);
                        }
                    }
                    Y_ref[((n * C + c) * H_out + oh) * W_out + ow] = sum;
                }
            }
        }
    }

    fn(X_data.data(), Y_result.data());

    for (int i = 0; i < total_y; ++i) {
        EXPECT_NEAR(static_cast<float>(Y_result[i]), Y_ref[i], 5e-2f) << "Mismatch at index " << i;
    }

    dlclose(h);
}


TEST(PoolingTest, MaxPool2D_QANT_Batched) {
    // Test batched 2D max pooling via QANT: X[4, 2, 6, 6] -> Y[4, 2, 3, 3]
    // kernel=2x2, stride=2, no padding
    builder::StructuredSDFGBuilder builder("maxpool2d_qant_batched", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::BFloat);
    types::Pointer desc_ptr(desc);

    builder.add_container("x", desc_ptr, true);
    builder.add_container("y", desc_ptr, true);

    auto& block = builder.add_block(sdfg.root());

    auto& x_node = builder.add_access(block, "x");
    auto& y_node = builder.add_access(block, "y");

    const int N = 4, C = 2, H = 6, W = 6;
    const int kH = 2, kW = 2;
    const int stride_val = 2, padding_val = 0;
    const int H_out = (H + 2 * padding_val - kH) / stride_val + 1;
    const int W_out = (W + 2 * padding_val - kW) / stride_val + 1;

    std::vector<symbolic::Expression> input_shape = {
        symbolic::integer(N), symbolic::integer(C), symbolic::integer(H), symbolic::integer(W)
    };
    std::vector<symbolic::Expression> kernel_shape = {symbolic::integer(kH), symbolic::integer(kW)};
    std::vector<symbolic::Expression> strides_vec = {symbolic::integer(stride_val), symbolic::integer(stride_val)};
    std::vector<symbolic::Expression> pads_vec = {
        symbolic::integer(padding_val),
        symbolic::integer(padding_val),
        symbolic::integer(padding_val),
        symbolic::integer(padding_val)
    };
    std::vector<symbolic::Expression> dilations_vec = {symbolic::integer(1), symbolic::integer(1)};

    types::Tensor input_tensor_x(desc.primitive_type(), input_shape);
    symbolic::MultiExpression output_shape = {
        symbolic::integer(N), symbolic::integer(C), symbolic::integer(H_out), symbolic::integer(W_out)
    };
    types::Tensor output_tensor(desc.primitive_type(), output_shape);

    auto& pooling_node = static_cast<math::tensor::PoolingNode&>(builder.add_library_node<math::tensor::PoolingNode>(
        block, DebugInfo(), math::tensor::PoolingMode::Max, input_shape, kernel_shape, strides_vec, pads_vec, dilations_vec
    ));

    builder.add_computational_memlet(block, x_node, pooling_node, "X", {}, input_tensor_x, block.debug_info());
    builder.add_computational_memlet(block, y_node, pooling_node, "Y", {}, output_tensor, block.debug_info());

    sdfg.validate();

    analysis::AnalysisManager analysis_manager(sdfg);

    sdfg::passes::Pipeline remapping("QantRemapping");
    remapping.register_pass<sdfg::passes::QantRemappingPass>();
    remapping.run(builder, analysis_manager);

    sdfg.validate();

    sdfg::passes::TensorToPointerConversionPass tensor_to_pointer_conversion_pass;
    tensor_to_pointer_conversion_pass.run(builder, analysis_manager);

    sdfg.validate();

    docc::qant::schedule(sdfg, "qant");

    sdfg.validate();

    std::string lib_path = docc::qant::so_compile(sdfg);

    void* h = dlopen(lib_path.c_str(), RTLD_LAZY);
    ASSERT_NE(h, nullptr) << dlerror();

    using Fn = void (*)(__bf16*, __bf16*);
    auto fn = reinterpret_cast<Fn>(dlsym(h, sdfg.name().c_str()));
    ASSERT_NE(fn, nullptr) << dlerror();

    const int total_x = N * C * H * W;
    const int total_y = N * C * H_out * W_out;

    std::vector<__bf16> X_data(total_x), Y_result(total_y, (__bf16) 0.0f);
    std::vector<float> Y_ref(total_y, 0.0f);

    for (int i = 0; i < total_x; ++i) X_data[i] = (__bf16) (static_cast<float>(i % 37) * 0.1f);

    // Reference max pooling
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    float max_val = -1e30f;
                    for (int kh_i = 0; kh_i < kH; ++kh_i) {
                        for (int kw_i = 0; kw_i < kW; ++kw_i) {
                            int ih = oh * stride_val + kh_i;
                            int iw = ow * stride_val + kw_i;
                            float x_val = static_cast<float>(X_data[((n * C + c) * H + ih) * W + iw]);
                            if (x_val > max_val) max_val = x_val;
                        }
                    }
                    Y_ref[((n * C + c) * H_out + oh) * W_out + ow] = max_val;
                }
            }
        }
    }

    fn(X_data.data(), Y_result.data());

    for (int i = 0; i < total_y; ++i) {
        EXPECT_NEAR(static_cast<float>(Y_result[i]), Y_ref[i], 5e-2f) << "Mismatch at index " << i;
    }

    dlclose(h);
}


TEST(PoolingTest, AvgPool2D_QANT_Batched) {
    // Test batched 2D average pooling via QANT: X[4, 2, 6, 6] -> Y[4, 2, 3, 3]
    // kernel=2x2, stride=2, no padding, mode=Sum
    builder::StructuredSDFGBuilder builder("avgpool2d_qant_batched", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::BFloat);
    types::Pointer desc_ptr(desc);

    builder.add_container("x", desc_ptr, true);
    builder.add_container("y", desc_ptr, true);

    auto& block = builder.add_block(sdfg.root());

    auto& x_node = builder.add_access(block, "x");
    auto& y_node = builder.add_access(block, "y");

    const int N = 4, C = 2, H = 6, W = 6;
    const int kH = 2, kW = 2;
    const int stride_val = 2, padding_val = 0;
    const int H_out = (H + 2 * padding_val - kH) / stride_val + 1;
    const int W_out = (W + 2 * padding_val - kW) / stride_val + 1;

    std::vector<symbolic::Expression> input_shape = {
        symbolic::integer(N), symbolic::integer(C), symbolic::integer(H), symbolic::integer(W)
    };
    std::vector<symbolic::Expression> kernel_shape = {symbolic::integer(kH), symbolic::integer(kW)};
    std::vector<symbolic::Expression> strides_vec = {symbolic::integer(stride_val), symbolic::integer(stride_val)};
    std::vector<symbolic::Expression> pads_vec = {
        symbolic::integer(padding_val),
        symbolic::integer(padding_val),
        symbolic::integer(padding_val),
        symbolic::integer(padding_val)
    };
    std::vector<symbolic::Expression> dilations_vec = {symbolic::integer(1), symbolic::integer(1)};

    types::Tensor input_tensor_x(desc.primitive_type(), input_shape);
    symbolic::MultiExpression output_shape = {
        symbolic::integer(N), symbolic::integer(C), symbolic::integer(H_out), symbolic::integer(W_out)
    };
    types::Tensor output_tensor(desc.primitive_type(), output_shape);

    auto& pooling_node = static_cast<math::tensor::PoolingNode&>(builder.add_library_node<math::tensor::PoolingNode>(
        block, DebugInfo(), math::tensor::PoolingMode::Sum, input_shape, kernel_shape, strides_vec, pads_vec, dilations_vec
    ));

    builder.add_computational_memlet(block, x_node, pooling_node, "X", {}, input_tensor_x, block.debug_info());
    builder.add_computational_memlet(block, y_node, pooling_node, "Y", {}, output_tensor, block.debug_info());

    sdfg.validate();

    analysis::AnalysisManager analysis_manager(sdfg);

    sdfg::passes::Pipeline remapping("QantRemapping");
    remapping.register_pass<sdfg::passes::QantRemappingPass>();
    remapping.run(builder, analysis_manager);

    sdfg.validate();

    sdfg::passes::TensorToPointerConversionPass tensor_to_pointer_conversion_pass;
    tensor_to_pointer_conversion_pass.run(builder, analysis_manager);

    sdfg.validate();

    docc::qant::schedule(sdfg, "qant");

    sdfg.validate();

    std::string lib_path = docc::qant::so_compile(sdfg);

    void* h = dlopen(lib_path.c_str(), RTLD_LAZY);
    ASSERT_NE(h, nullptr) << dlerror();

    using Fn = void (*)(__bf16*, __bf16*);
    auto fn = reinterpret_cast<Fn>(dlsym(h, sdfg.name().c_str()));
    ASSERT_NE(fn, nullptr) << dlerror();

    const int total_x = N * C * H * W;
    const int total_y = N * C * H_out * W_out;

    std::vector<__bf16> X_data(total_x), Y_result(total_y, (__bf16) 0.0f);
    std::vector<float> Y_ref(total_y, 0.0f);

    for (int i = 0; i < total_x; ++i) X_data[i] = (__bf16) (static_cast<float>(i % 37) * 0.1f);

    // Reference sum pooling
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    float sum = 0.0f;
                    for (int kh_i = 0; kh_i < kH; ++kh_i) {
                        for (int kw_i = 0; kw_i < kW; ++kw_i) {
                            int ih = oh * stride_val + kh_i;
                            int iw = ow * stride_val + kw_i;
                            sum += static_cast<float>(X_data[((n * C + c) * H + ih) * W + iw]);
                        }
                    }
                    Y_ref[((n * C + c) * H_out + oh) * W_out + ow] = sum;
                }
            }
        }
    }

    fn(X_data.data(), Y_result.data());

    for (int i = 0; i < total_y; ++i) {
        EXPECT_NEAR(static_cast<float>(Y_result[i]), Y_ref[i], 5e-2f) << "Mismatch at index " << i;
    }

    dlclose(h);
}
