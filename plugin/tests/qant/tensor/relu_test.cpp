#include <gtest/gtest.h>

#include <dlfcn.h>
#include <stdfloat>
#include <vector>

#include <sdfg/passes/dataflow/tensor_to_pointer_conversion.h>
#include <sdfg/passes/targets/target_mapping_pass.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/relu_node.h"
#include "sdfg/passes/pipeline.h"
#include "sdfg_debug_dump.h"
#include "so_compile.h"

#include "docc/qant/passes/remapping_pass.h"
#include "docc/qant/plugin.h"
#include "docc/qant/qant.h"

using namespace sdfg;

TEST(ReLUTest, ReLU_1D_Vector) {
    // Test simple 1D vector ReLU: Y = max(0, X) for X[N]
    builder::StructuredSDFGBuilder builder("relu_1d", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::BFloat);
    types::Pointer desc_ptr(desc);

    builder.add_container("x", desc_ptr, true);
    builder.add_container("y", desc_ptr, true);

    auto& block = builder.add_block(sdfg.root());

    auto& x_node = builder.add_access(block, "x");
    auto& y_node = builder.add_access(block, "y");

    // Shape: [16] (simple 1D vector)
    symbolic::MultiExpression shape = {symbolic::integer(16)};

    types::Tensor input_tensor(desc.primitive_type(), shape);
    types::Tensor output_tensor(desc.primitive_type(), shape);

    auto& relu_node =
        static_cast<math::tensor::ReLUNode&>(builder.add_library_node<math::tensor::ReLUNode>(block, DebugInfo(), shape)
        );

    builder.add_computational_memlet(block, x_node, relu_node, "X", {}, input_tensor, block.debug_info());
    builder.add_computational_memlet(block, y_node, relu_node, "Y", {}, output_tensor, block.debug_info());

    // Check basic properties
    EXPECT_EQ(relu_node.inputs().size(), 2);
    EXPECT_EQ(relu_node.input(0), "Y");
    EXPECT_EQ(relu_node.input(1), "X");
    EXPECT_EQ(relu_node.outputs().size(), 0);

    sdfg.validate();

    EXPECT_EQ(block.dataflow().nodes().size(), 3); // x, y, relu

    sdfg.validate();

    dump_sdfg(sdfg, "0.before-remap");

    analysis::AnalysisManager analysis_manager(sdfg);

    // Run expansion pass
    sdfg::passes::Pipeline expansion("QantRemapping");
    expansion.register_pass<sdfg::passes::QantRemappingPass>();
    expansion.run(builder, analysis_manager);

    dump_sdfg(sdfg, "1.after-remap");

    sdfg.validate();

    auto library_nodes = block.dataflow().library_nodes();
    EXPECT_EQ(library_nodes.size(), 1);
    auto* new_node = dynamic_cast<sdfg::math::tensor::ReLUNode*>(*library_nodes.begin());
    EXPECT_TRUE(new_node);
    EXPECT_EQ(new_node->implementation_type(), docc::qant::ImplementationType_QANT.value());

    sdfg::passes::TensorToPointerConversionPass tensor_to_pointer_conversion_pass;
    tensor_to_pointer_conversion_pass.run(builder, analysis_manager);

    dump_sdfg(sdfg, "2.after-t2p");

    sdfg.validate();

    docc::qant::schedule(sdfg, "qant");

    dump_sdfg(sdfg, "3.after-sched");

    sdfg.validate();

    std::string lib_path = docc::qant::so_compile(sdfg);

    void* h = dlopen(lib_path.c_str(), RTLD_LAZY);
    ASSERT_NE(h, nullptr) << dlerror();

    using Fn = void (*)(__bf16*, __bf16*);
    auto fn = reinterpret_cast<Fn>(dlsym(h, sdfg.name().c_str()));
    ASSERT_NE(fn, nullptr) << dlerror();

    const int N = 16;
    std::vector<__bf16> X(N), Y_result(N, (__bf16) 0.0f);
    std::vector<float> Y_ref(N, 0.0f);

    // Initialize with some negative and positive values
    for (int i = 0; i < N; ++i) {
        X[i] = (__bf16) (static_cast<float>(i - 8) * 0.5f); // Range: -4.0 to 3.5
    }

    // Compute reference ReLU
    for (int i = 0; i < N; ++i) {
        float val = static_cast<float>(X[i]);
        Y_ref[i] = (val > 0.0f) ? val : 0.0f;
    }

    fn(X.data(), Y_result.data());

    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(static_cast<float>(Y_result[i]), Y_ref[i], 1e-3f) << "Mismatch at index " << i;
    }

    dlclose(h);
}

TEST(ReLUTest, ReLU_2D_Matrix) {
    // Test 2D matrix ReLU: Y = max(0, X) for X[M, N]
    builder::StructuredSDFGBuilder builder("relu_2d", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::BFloat);
    types::Pointer desc_ptr(desc);

    builder.add_container("x", desc_ptr, true);
    builder.add_container("y", desc_ptr, true);

    auto& block = builder.add_block(sdfg.root());

    auto& x_node = builder.add_access(block, "x");
    auto& y_node = builder.add_access(block, "y");

    // Shape: [4, 8] (4 rows, 8 columns)
    symbolic::MultiExpression shape = {symbolic::integer(4), symbolic::integer(8)};

    types::Tensor input_tensor(desc.primitive_type(), shape);
    types::Tensor output_tensor(desc.primitive_type(), shape);

    auto& relu_node =
        static_cast<math::tensor::ReLUNode&>(builder.add_library_node<math::tensor::ReLUNode>(block, DebugInfo(), shape)
        );

    builder.add_computational_memlet(block, x_node, relu_node, "X", {}, input_tensor, block.debug_info());
    builder.add_computational_memlet(block, y_node, relu_node, "Y", {}, output_tensor, block.debug_info());

    sdfg.validate();

    dump_sdfg(sdfg, "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);

    // Run expansion pass
    sdfg::passes::Pipeline expansion("QantRemapping");
    expansion.register_pass<sdfg::passes::QantRemappingPass>();
    expansion.run(builder, analysis_manager);

    dump_sdfg(sdfg, "1.expand");

    sdfg.validate();

    auto library_nodes = block.dataflow().library_nodes();
    EXPECT_EQ(library_nodes.size(), 1);
    auto* new_node = dynamic_cast<sdfg::math::tensor::ReLUNode*>(*library_nodes.begin());
    EXPECT_TRUE(new_node);
    EXPECT_EQ(new_node->implementation_type(), docc::qant::ImplementationType_QANT.value());

    sdfg::passes::TensorToPointerConversionPass tensor_to_pointer_conversion_pass;
    tensor_to_pointer_conversion_pass.run(builder, analysis_manager);

    dump_sdfg(sdfg, "2.t2p");

    sdfg.validate();

    docc::qant::schedule(sdfg, "qant");

    dump_sdfg(sdfg, "3.sched");

    sdfg.validate();

    std::string lib_path = docc::qant::so_compile(sdfg);

    void* h = dlopen(lib_path.c_str(), RTLD_LAZY);
    ASSERT_NE(h, nullptr) << dlerror();

    using Fn = void (*)(__bf16*, __bf16*);
    auto fn = reinterpret_cast<Fn>(dlsym(h, sdfg.name().c_str()));
    ASSERT_NE(fn, nullptr) << dlerror();

    const int M = 4, N = 8;
    const int total = M * N;
    std::vector<__bf16> X(total), Y_result(total, (__bf16) 0.0f);
    std::vector<float> Y_ref(total, 0.0f);

    // Initialize with mixed positive and negative values
    for (int i = 0; i < total; ++i) {
        X[i] = (__bf16) (static_cast<float>(i - 16) * 0.25f); // Range: -4.0 to 3.75
    }

    // Compute reference ReLU
    for (int i = 0; i < total; ++i) {
        float val = static_cast<float>(X[i]);
        Y_ref[i] = (val > 0.0f) ? val : 0.0f;
    }

    fn(X.data(), Y_result.data());

    for (int i = 0; i < total; ++i) {
        EXPECT_NEAR(static_cast<float>(Y_result[i]), Y_ref[i], 1e-3f) << "Mismatch at index " << i;
    }

    dlclose(h);
}
