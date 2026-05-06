#include <gtest/gtest.h>

#include <dlfcn.h>
#include <vector>

#include <sdfg/passes/dataflow/tensor_to_pointer_conversion.h>
#include <sdfg/passes/targets/target_mapping_pass.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/passes/pipeline.h"
#include "sdfg_debug_dump.h"
#include "so_compile.h"

#include "docc/qant/dataflow/library_nodes/math/tensor/matmul_node.h"
#include "docc/qant/passes/remapping_pass.h"
#include "docc/qant/plugin.h"
#include "docc/qant/qant.h"

using namespace sdfg;

TEST(MatMulTest, MatMul_2D_SimpleMatrix) {
    // Test simple 2D matrix multiplication: A[M, K] @ B[K, N] = Y[M, N]
    builder::StructuredSDFGBuilder builder("matmul_2d", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::BFloat);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr, true);
    builder.add_container("b", desc_ptr, true);
    builder.add_container("y", desc_ptr, true);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& y_node = builder.add_access(block, "y");

    // Shape A: [4, 8] (M=4, K=8)
    // Shape B: [8, 6] (K=8, N=6)
    // Output Y: [4, 6] (M=4, N=6)
    symbolic::MultiExpression shape_a = {symbolic::integer(4), symbolic::integer(8)};
    symbolic::MultiExpression shape_b = {symbolic::integer(8), symbolic::integer(6)};


    types::Tensor input_tensor_a(desc.primitive_type(), shape_a);
    types::Tensor input_tensor_b(desc.primitive_type(), shape_b);
    symbolic::MultiExpression output_shape = {symbolic::integer(4), symbolic::integer(6)};
    types::Tensor output_tensor(desc.primitive_type(), output_shape);

    auto& matmul_node =
        static_cast<math::tensor::MatMulNode&>(builder.add_library_node<
                                               math::tensor::MatMulNode>(block, DebugInfo(), shape_a, shape_b));

    builder.add_computational_memlet(block, a_node, matmul_node, "A", {}, input_tensor_a, block.debug_info());
    builder.add_computational_memlet(block, b_node, matmul_node, "B", {}, input_tensor_b, block.debug_info());
    builder.add_computational_memlet(block, matmul_node, "Y", y_node, {}, output_tensor, block.debug_info());

    // Check basic properties
    EXPECT_EQ(matmul_node.inputs().size(), 2);
    EXPECT_EQ(matmul_node.inputs()[0], "A");
    EXPECT_EQ(matmul_node.inputs()[1], "B");
    EXPECT_EQ(matmul_node.outputs().size(), 1);
    EXPECT_EQ(matmul_node.outputs()[0], "Y");

    // Check dimensions
    EXPECT_TRUE(symbolic::eq(matmul_node.m(), symbolic::integer(4)));
    EXPECT_TRUE(symbolic::eq(matmul_node.n(), symbolic::integer(6)));
    EXPECT_TRUE(symbolic::eq(matmul_node.k(), symbolic::integer(8)));

    sdfg.validate();

    EXPECT_EQ(block.dataflow().nodes().size(), 4); // a, b, y, matmul

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
    auto* new_node = dynamic_cast<sdfg::math::tensor::QantMatMulNode*>(*library_nodes.begin());
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

    using Fn = void (*)(__bf16*, __bf16*, __bf16*);
    auto fn = reinterpret_cast<Fn>(dlsym(h, sdfg.name().c_str()));
    ASSERT_NE(fn, nullptr) << dlerror();

    const int M = 4, K = 8, N = 6;
    std::vector<__bf16> A(M * K), B(K * N), Y_result(M * N, (__bf16) 0.0f);
    std::vector<float> Y_ref(M * N, 0.0f);
    for (int i = 0; i < M * K; ++i) A[i] = (__bf16) (static_cast<float>(i % 7) * 0.5f);
    for (int i = 0; i < K * N; ++i) B[i] = (__bf16) (static_cast<float>(i % 5) * 0.3f);

    // Compute reference matmul in float
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < K; ++p) {
                sum += static_cast<float>(A[i * K + p]) * static_cast<float>(B[p * N + j]);
            }
            Y_ref[i * N + j] = sum;
        }
    }

    fn(A.data(), B.data(), Y_result.data());

    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(static_cast<float>(Y_result[i]), Y_ref[i], 3e-2f) << "Mismatch at index " << i;
    }

    dlclose(h);
}

TEST(MatMulTest, MatMul_3D_Batched) {
    // Test batched matrix multiplication: A[B, M, K] @ B[B, K, N] = Y[B, M, N]
    builder::StructuredSDFGBuilder builder("matmul_3d", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::BFloat);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr, true);
    builder.add_container("b", desc_ptr, true);
    builder.add_container("y", desc_ptr, true);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& y_node = builder.add_access(block, "y");

    // Shape A: [2, 4, 8] (B=2, M=4, K=8)
    // Shape B: [2, 8, 6] (B=2, K=8, N=6)
    // Output Y: [2, 4, 6] (B=2, M=4, N=6)
    symbolic::MultiExpression shape_a = {symbolic::integer(2), symbolic::integer(4), symbolic::integer(8)};
    symbolic::MultiExpression shape_b = {symbolic::integer(2), symbolic::integer(8), symbolic::integer(6)};

    types::Tensor input_tensor_a(desc.primitive_type(), shape_a);
    types::Tensor input_tensor_b(desc.primitive_type(), shape_b);
    symbolic::MultiExpression output_shape = {symbolic::integer(2), symbolic::integer(4), symbolic::integer(6)};
    types::Tensor output_tensor(desc.primitive_type(), output_shape);

    auto& matmul_node =
        static_cast<math::tensor::MatMulNode&>(builder.add_library_node<
                                               math::tensor::MatMulNode>(block, DebugInfo(), shape_a, shape_b));

    builder.add_computational_memlet(block, a_node, matmul_node, "A", {}, input_tensor_a, block.debug_info());
    builder.add_computational_memlet(block, b_node, matmul_node, "B", {}, input_tensor_b, block.debug_info());
    builder.add_computational_memlet(block, matmul_node, "Y", y_node, {}, output_tensor, block.debug_info());

    // Check dimensions
    EXPECT_TRUE(symbolic::eq(matmul_node.m(), symbolic::integer(4)));
    EXPECT_TRUE(symbolic::eq(matmul_node.n(), symbolic::integer(6)));
    EXPECT_TRUE(symbolic::eq(matmul_node.k(), symbolic::integer(8)));

    dump_sdfg(sdfg, "0.before");

    analysis::AnalysisManager analysis_manager_3d(sdfg);

    sdfg.validate();

    // Run expansion pass
    sdfg::passes::Pipeline expansion("QantRemapping");
    expansion.register_pass<sdfg::passes::QantRemappingPass>();
    expansion.run(builder, analysis_manager_3d);

    dump_sdfg(sdfg, "1.expand");

    sdfg.validate();

    auto library_nodes = block.dataflow().library_nodes();
    EXPECT_EQ(library_nodes.size(), 1);
    auto* new_node = dynamic_cast<sdfg::math::tensor::QantMatMulNode*>(*library_nodes.begin());
    EXPECT_TRUE(new_node);
    EXPECT_EQ(new_node->implementation_type(), docc::qant::ImplementationType_QANT.value());

    sdfg::passes::TensorToPointerConversionPass tensor_to_pointer_conversion_pass;
    tensor_to_pointer_conversion_pass.run(builder, analysis_manager_3d);

    dump_sdfg(sdfg, "2.t2p");

    sdfg.validate();

    docc::qant::schedule(sdfg, "qant");

    dump_sdfg(sdfg, "3.sched");

    sdfg.validate();

    std::string lib_path = docc::qant::so_compile(sdfg);

    void* h = dlopen(lib_path.c_str(), RTLD_LAZY);
    ASSERT_NE(h, nullptr) << dlerror();

    using Fn = void (*)(__bf16*, __bf16*, __bf16*);
    auto fn = reinterpret_cast<Fn>(dlsym(h, sdfg.name().c_str()));
    ASSERT_NE(fn, nullptr) << dlerror();

    const int Batch = 2, M = 4, K = 8, N = 6;
    const int total_a = Batch * M * K, total_b = Batch * K * N, total_y = Batch * M * N;
    std::vector<__bf16> A(total_a), B(total_b), Y_result(total_y, (__bf16) 0.0f);
    std::vector<float> Y_ref(total_y, 0.0f);
    for (int i = 0; i < total_a; ++i) A[i] = (__bf16) (static_cast<float>(i % 7) * 0.5f);
    for (int i = 0; i < total_b; ++i) B[i] = (__bf16) (static_cast<float>(i % 5) * 0.3f);

    // Compute reference batched matmul in float
    for (int batch = 0; batch < Batch; ++batch) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int p = 0; p < K; ++p) {
                    sum += static_cast<float>(A[batch * M * K + i * K + p]) *
                           static_cast<float>(B[batch * K * N + p * N + j]);
                }
                Y_ref[batch * M * N + i * N + j] = sum;
            }
        }
    }

    fn(A.data(), B.data(), Y_result.data());

    for (int i = 0; i < total_y; ++i) {
        EXPECT_NEAR(static_cast<float>(Y_result[i]), Y_ref[i], 3e-2f) << "Mismatch at index " << i;
    }

    dlclose(h);
}
