#include <dlfcn.h>
#include <gtest/gtest.h>
#include <nlohmann/json_fwd.hpp>
#include <stdfloat>
#include <string>
#include <vector>

#include <sdfg/passes/dataflow/tensor_to_pointer_conversion.h>
#include <sdfg/passes/pipeline.h>
#include "docc/qant/dataflow/library_nodes/math/tensor/matmul_node.h"
#include "docc/qant/passes/remapping_pass.h"
#include "docc/qant/qant.h"
#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/einsum_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"
#include "sdfg_debug_dump.h"
#include "so_compile.h"

#include "docc/qant/plugin.h"
#include "docc/qant/qant.h"

using namespace sdfg;

TEST(EinsumTest, Matmul_2D_BFloat16) {
    // Test simple 2D matrix multiplication: A[M, K] @ B[K, N] = Y[M, N]
    builder::StructuredSDFGBuilder builder("einsum_matmul", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::BFloat);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr, true);
    builder.add_container("b", desc_ptr, true);
    builder.add_container("c", desc_ptr, true);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& c_node = builder.add_access(block, "c");
    auto& c_out_node = builder.add_access(block, "c");

    symbolic::MultiExpression shape_a = {symbolic::integer(4), symbolic::integer(8)};
    symbolic::MultiExpression shape_b = {symbolic::integer(8), symbolic::integer(6)};


    types::Tensor input_tensor_a(desc.primitive_type(), shape_a);
    types::Tensor input_tensor_b(desc.primitive_type(), shape_b);
    symbolic::MultiExpression output_shape = {symbolic::integer(4), symbolic::integer(6)};
    types::Tensor output_tensor(desc.primitive_type(), output_shape);


    // Symbols
    auto zero = symbolic::zero();
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto l = symbolic::symbol("l");
    auto m = symbolic::symbol("m");
    auto n = symbolic::symbol("n");

    auto& libnode = builder.add_library_node<
        math::tensor::EinsumNode,
        const std::vector<std::string>&,
        const std::vector<math::tensor::EinsumDimension>&,
        const data_flow::Subset&,
        const std::vector<data_flow::Subset>&>(
        block, DebugInfo(), {"_in1", "_in2"}, {{i, zero, l}, {j, zero, m}, {k, zero, n}}, {i, j}, {{i, k}, {k, j}}, false
    );

    builder.add_computational_memlet(block, a_node, libnode, "_in1", {}, input_tensor_a, block.debug_info());
    builder.add_computational_memlet(block, b_node, libnode, "_in2", {}, input_tensor_b, block.debug_info());
    builder.add_computational_memlet(block, c_node, libnode, "__einsum_out", {}, output_tensor, block.debug_info());
    builder.add_computational_memlet(block, libnode, "__einsum_out", c_out_node, {}, output_tensor, block.debug_info());

    sdfg.validate();

    // Validate einsum node properties
    auto library_nodes = block.dataflow().library_nodes();
    EXPECT_EQ(library_nodes.size(), 1);
    auto* einsum_node = dynamic_cast<sdfg::math::tensor::EinsumNode*>(*library_nodes.begin());
    EXPECT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->dims().size(), 3);
    ASSERT_GE(einsum_node->dims().size(), 3);
    EXPECT_TRUE(symbolic::eq(einsum_node->indvar(0), i));
    EXPECT_TRUE(symbolic::eq(einsum_node->init(0), zero));
    EXPECT_TRUE(symbolic::eq(einsum_node->bound(0), l));
    EXPECT_TRUE(symbolic::eq(einsum_node->indvar(1), j));
    EXPECT_TRUE(symbolic::eq(einsum_node->init(1), zero));
    EXPECT_TRUE(symbolic::eq(einsum_node->bound(1), m));
    EXPECT_TRUE(symbolic::eq(einsum_node->indvar(2), k));
    EXPECT_TRUE(symbolic::eq(einsum_node->init(2), zero));
    EXPECT_TRUE(symbolic::eq(einsum_node->bound(2), n));

    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"__einsum_out"}));
    EXPECT_EQ(einsum_node->out_indices().size(), 2);
    ASSERT_GE(einsum_node->out_indices().size(), 2);
    EXPECT_TRUE(symbolic::eq(einsum_node->out_index(0), i));
    EXPECT_TRUE(symbolic::eq(einsum_node->out_index(1), j));

    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in1", "_in2", "__einsum_out"}));
    EXPECT_EQ(einsum_node->in_indices().size(), 3);
    ASSERT_GE(einsum_node->in_indices().size(), 3);
    EXPECT_EQ(einsum_node->in_indices(0).size(), 2);
    ASSERT_GE(einsum_node->in_indices(0).size(), 2);
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(0, 0), i));
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(0, 1), k));
    EXPECT_EQ(einsum_node->in_indices(1).size(), 2);
    ASSERT_GE(einsum_node->in_indices(1).size(), 2);
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(1, 0), k));
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(1, 1), j));
    EXPECT_EQ(einsum_node->in_indices(2).size(), 2);
    ASSERT_GE(einsum_node->in_indices(2).size(), 2);
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(2, 0), i));
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(2, 1), j));

    auto symbols = einsum_node->symbols();
    EXPECT_EQ(symbols.size(), 3);
    EXPECT_TRUE(symbols.contains(l));
    EXPECT_TRUE(symbols.contains(m));
    EXPECT_TRUE(symbols.contains(n));

    EXPECT_EQ(
        einsum_node->toStr(),
        "__einsum_out[i][j] = _in1[i][k] * _in2[k][j] + __einsum_out[i][j] for i = 0 : l for j = 0 : m for k = 0 : n"
    );

    EXPECT_TRUE(symbolic::eq(einsum_node->flop(), SymEngine::mul({l, m, n, symbolic::integer(2)})));

    EXPECT_NO_THROW(sdfg.validate());

    // Test Einsum2QantMatmul transformation
    analysis::AnalysisManager analysis_manager(sdfg);

    // Run expansion pass
    sdfg::passes::Pipeline expansion("QantRemapping");
    expansion.register_pass<sdfg::passes::QantRemappingPass>();
    expansion.run(builder, analysis_manager);

    sdfg.validate();

    // Verify the einsum node has been replaced with a QantMatMulNode
    library_nodes = block.dataflow().library_nodes();
    EXPECT_EQ(library_nodes.size(), 1);
    auto* qant_matmul = dynamic_cast<sdfg::math::tensor::QantMatMulNode*>(*library_nodes.begin());
    EXPECT_TRUE(qant_matmul);
    EXPECT_EQ(qant_matmul->code(), sdfg::math::tensor::LibraryNodeType_QantMatMul);
    EXPECT_EQ(qant_matmul->quantization(), types::PrimitiveType::BFloat);
    EXPECT_EQ(qant_matmul->implementation_type(), docc::qant::ImplementationType_QANT.value());

    // Verify the node has correct inputs and outputs
    EXPECT_EQ(block.dataflow().in_degree(*qant_matmul), 2); // A and B inputs
    EXPECT_EQ(block.dataflow().out_degree(*qant_matmul), 1); // Y output

    sdfg::passes::TensorToPointerConversionPass tensor_to_pointer_conversion_pass;
    tensor_to_pointer_conversion_pass.run(builder, analysis_manager);

    sdfg.validate();

    docc::qant::schedule(sdfg, "qant");

    sdfg.validate();

    // Compile the SDFG and test execution
    std::string lib_path = docc::qant::so_compile(sdfg);

    void* h = dlopen(lib_path.c_str(), RTLD_LAZY);
    ASSERT_NE(h, nullptr) << dlerror();

    using Fn = void (*)(__bf16*, __bf16*, __bf16*);
    auto fn = reinterpret_cast<Fn>(dlsym(h, sdfg.name().c_str()));
    ASSERT_NE(fn, nullptr) << dlerror();

    const int M = 4, K = 8, N = 6;
    std::vector<__bf16> A(M * K), B(K * N), C_result(M * N, (__bf16) 0.0f);
    std::vector<float> C_ref(M * N, 0.0f);

    // Initialize input data
    for (int i = 0; i < M * K; ++i) A[i] = (__bf16) (static_cast<float>(i % 7) * 0.5f);
    for (int i = 0; i < K * N; ++i) B[i] = (__bf16) (static_cast<float>(i % 5) * 0.3f);

    // Compute reference matmul in float
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < K; ++p) {
                sum += static_cast<float>(A[i * K + p]) * static_cast<float>(B[p * N + j]);
            }
            C_ref[i * N + j] = sum;
        }
    }

    // Execute compiled function
    fn(A.data(), B.data(), C_result.data());

    // Verify results match reference computation
    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(static_cast<float>(C_result[i]), C_ref[i], 3e-2f) << "Mismatch at index " << i;
    }

    dlclose(h);
}


TEST(EinsumTest, Transpose_BFloat16) {
    // Test simple 2D matrix multiplication: A[M, K] @ B[K, N] = Y[M, N]
    builder::StructuredSDFGBuilder builder("einsum_transpose", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::BFloat);
    types::Scalar desc_int(types::PrimitiveType::Int64);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr, true);
    builder.add_container("c", desc_ptr, true);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& c_node = builder.add_access(block, "c");
    auto& c_out_node = builder.add_access(block, "c");

    symbolic::MultiExpression shape_a = {symbolic::integer(4), symbolic::integer(8)};


    types::Tensor input_tensor_a(desc.primitive_type(), shape_a);
    symbolic::MultiExpression output_shape = {symbolic::integer(8), symbolic::integer(4)};
    types::Tensor output_tensor(desc.primitive_type(), output_shape);

    auto zero = symbolic::zero();
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    builder.add_container("i", desc_int);
    builder.add_container("j", desc_int);

    auto& libnode = builder.add_library_node<
        math::tensor::EinsumNode,
        const std::vector<std::string>&,
        const std::vector<math::tensor::EinsumDimension>&,
        const data_flow::Subset&,
        const std::vector<data_flow::Subset>&>(
        block,
        DebugInfo(),
        {"_in1"},
        {{i, zero, symbolic::integer(8)}, {j, zero, symbolic::integer(4)}},
        {i, j},
        {{j, i}},
        false
    );

    builder.add_computational_memlet(block, a_node, libnode, "_in1", {}, input_tensor_a, block.debug_info());
    builder.add_computational_memlet(block, c_node, libnode, "__einsum_out", {}, output_tensor, block.debug_info());
    builder.add_computational_memlet(block, libnode, "__einsum_out", c_out_node, {}, output_tensor, block.debug_info());

    sdfg.validate();

    auto library_nodes = block.dataflow().library_nodes();
    EXPECT_EQ(library_nodes.size(), 1);
    auto* einsum_node = dynamic_cast<sdfg::math::tensor::EinsumNode*>(*library_nodes.begin());
    EXPECT_TRUE(einsum_node);

    sdfg::analysis::AnalysisManager analysis_manager(builder.subject());

    sdfg::passes::Pipeline libnode_expansion = sdfg::passes::Pipeline::expansion();
    libnode_expansion.run(builder, analysis_manager);

    sdfg::passes::TensorToPointerConversionPass tensor_to_pointer_conversion_pass;
    tensor_to_pointer_conversion_pass.run(builder, analysis_manager);

    dump_sdfg(sdfg, "0.after_expansion");

    sdfg.validate();

    docc::qant::schedule(sdfg, "qant");

    sdfg.validate();

    // Compile the SDFG and test execution
    std::string lib_path = docc::qant::so_compile(sdfg);

    void* h = dlopen(lib_path.c_str(), RTLD_LAZY);
    ASSERT_NE(h, nullptr) << dlerror();

    using Fn = void (*)(__bf16*, __bf16*);
    auto fn = reinterpret_cast<Fn>(dlsym(h, sdfg.name().c_str()));
    ASSERT_NE(fn, nullptr) << dlerror();

    const int M_orig = 4, N_orig = 8;
    std::vector<__bf16> A_data(M_orig * N_orig);
    std::vector<__bf16> C_result(N_orig * M_orig, (__bf16) 0.0f);
    std::vector<float> C_ref(N_orig * M_orig, 0.0f);

    // Initialize input data: A[4, 8]
    for (int i = 0; i < M_orig * N_orig; ++i) {
        A_data[i] = (__bf16) (static_cast<float>(i % 7) * 0.5f);
    }

    // Compute reference transpose: C[8, 4] = A[4, 8]^T
    for (int i = 0; i < M_orig; ++i) {
        for (int j = 0; j < N_orig; ++j) {
            // A[i, j] goes to C[j, i]
            C_ref[j * M_orig + i] = static_cast<float>(A_data[i * N_orig + j]);
        }
    }

    // Execute compiled function
    fn(A_data.data(), C_result.data());

    // Verify results match reference computation
    for (int i = 0; i < N_orig * M_orig; ++i) {
        EXPECT_NEAR(static_cast<float>(C_result[i]), C_ref[i], 1e-6f) << "Mismatch at index " << i;
    }

    dlclose(h);
}

TEST(EinsumTest, GEMM_3D_Batched_BFloat16) {
    builder::StructuredSDFGBuilder builder("batched_matmul_3d", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::BFloat);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr, true);
    builder.add_container("b", desc_ptr, true);
    builder.add_container("c", desc_ptr, true);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& c_node = builder.add_access(block, "c");
    auto& c_out_node = builder.add_access(block, "c");

    symbolic::MultiExpression shape_a = {symbolic::integer(2), symbolic::integer(4), symbolic::integer(8)};
    symbolic::MultiExpression shape_b = {symbolic::integer(2), symbolic::integer(8), symbolic::integer(6)};

    types::Tensor input_tensor_a(desc.primitive_type(), shape_a);
    types::Tensor input_tensor_b(desc.primitive_type(), shape_b);
    symbolic::MultiExpression output_shape = {symbolic::integer(2), symbolic::integer(4), symbolic::integer(6)};
    types::Tensor output_tensor(desc.primitive_type(), output_shape);

    // Symbols for einsum
    auto zero = symbolic::zero();
    auto b = symbolic::symbol("b"); // batch dimension
    auto i = symbolic::symbol("i"); // M dimension
    auto j = symbolic::symbol("j"); // N dimension
    auto k = symbolic::symbol("k"); // K dimension (reduction)
    auto B = symbolic::symbol("B"); // batch size = 2
    auto M = symbolic::symbol("M"); // M = 4
    auto N = symbolic::symbol("N"); // N = 6
    auto K = symbolic::symbol("K"); // K = 8

    auto& libnode = builder.add_library_node<
        math::tensor::EinsumNode,
        const std::vector<std::string>&,
        const std::vector<math::tensor::EinsumDimension>&,
        const data_flow::Subset&,
        const std::vector<data_flow::Subset>&>(
        block,
        DebugInfo(),
        {"_in1", "_in2"},
        {{b, zero, symbolic::integer(2)},
         {i, zero, symbolic::integer(4)},
         {j, zero, symbolic::integer(6)},
         {k, zero, symbolic::integer(8)}},
        {b, i, j},
        {{b, i, k}, {b, k, j}},
        false
    );

    builder.add_computational_memlet(block, a_node, libnode, "_in1", {}, input_tensor_a, block.debug_info());
    builder.add_computational_memlet(block, b_node, libnode, "_in2", {}, input_tensor_b, block.debug_info());
    builder.add_computational_memlet(block, c_node, libnode, "__einsum_out", {}, output_tensor, block.debug_info());
    builder.add_computational_memlet(block, libnode, "__einsum_out", c_out_node, {}, output_tensor, block.debug_info());

    sdfg.validate();

    // Validate einsum node properties
    auto library_nodes = block.dataflow().library_nodes();
    EXPECT_EQ(library_nodes.size(), 1);
    auto* einsum_node = dynamic_cast<sdfg::math::tensor::EinsumNode*>(*library_nodes.begin());
    EXPECT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->dims().size(), 4); // 4 dimensions: b, i, j, k

    // Check output indices (batch, row, col)
    EXPECT_EQ(einsum_node->out_indices().size(), 3);
    ASSERT_GE(einsum_node->out_indices().size(), 3);
    EXPECT_TRUE(symbolic::eq(einsum_node->out_index(0), b));
    EXPECT_TRUE(symbolic::eq(einsum_node->out_index(1), i));
    EXPECT_TRUE(symbolic::eq(einsum_node->out_index(2), j));

    // Check inputs
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in1", "_in2", "__einsum_out"}));
    EXPECT_EQ(einsum_node->in_indices().size(), 3);

    // Input A: [b, i, k]
    EXPECT_EQ(einsum_node->in_indices(0).size(), 3);
    ASSERT_GE(einsum_node->in_indices(0).size(), 3);
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(0, 0), b));
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(0, 1), i));
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(0, 2), k));

    // Input B: [b, k, j]
    EXPECT_EQ(einsum_node->in_indices(1).size(), 3);
    ASSERT_GE(einsum_node->in_indices(1).size(), 3);
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(1, 0), b));
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(1, 1), k));
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(1, 2), j));

    // Output C: [b, i, j]
    EXPECT_EQ(einsum_node->in_indices(2).size(), 3);
    ASSERT_GE(einsum_node->in_indices(2).size(), 3);
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(2, 0), b));
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(2, 1), i));
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(2, 2), j));

    EXPECT_NO_THROW(sdfg.validate());

    // Test Einsum2QantMatmul transformation on batched matmul
    analysis::AnalysisManager analysis_manager(sdfg);

    // Run QantRemapping pass which should transform einsum to QantMatMul
    sdfg::passes::Pipeline expansion("QantRemapping");
    expansion.register_pass<sdfg::passes::QantRemappingPass>();
    expansion.run(builder, analysis_manager);

    sdfg.validate();

    // Verify the einsum node has been replaced with a QantMatMulNode
    library_nodes = block.dataflow().library_nodes();
    EXPECT_EQ(library_nodes.size(), 1);
    auto* qant_matmul = dynamic_cast<sdfg::math::tensor::QantMatMulNode*>(*library_nodes.begin());
    EXPECT_TRUE(qant_matmul);
    EXPECT_EQ(qant_matmul->code(), sdfg::math::tensor::LibraryNodeType_QantMatMul);
    EXPECT_EQ(qant_matmul->quantization(), types::PrimitiveType::BFloat);
    EXPECT_EQ(qant_matmul->implementation_type(), docc::qant::ImplementationType_QANT.value());

    // Verify the node has correct inputs and outputs
    EXPECT_EQ(block.dataflow().in_degree(*qant_matmul), 2); // A and B inputs
    EXPECT_EQ(block.dataflow().out_degree(*qant_matmul), 1); // Y output

    // Verify tensor layouts include batch dimension
    const auto& layout_a = qant_matmul->layout_a();
    const auto& layout_b = qant_matmul->layout_b();

    // Layout A should be [2, 4, 8] (Batch=2, M=4, K=8)
    EXPECT_EQ(layout_a.dims(), 3);
    ASSERT_GE(layout_a.shape().size(), 3);
    EXPECT_TRUE(symbolic::eq(layout_a.shape()[0], symbolic::integer(2))); // Batch
    EXPECT_TRUE(symbolic::eq(layout_a.shape()[1], symbolic::integer(4))); // M
    EXPECT_TRUE(symbolic::eq(layout_a.shape()[2], symbolic::integer(8))); // K

    // Layout B should be [2, 8, 6] (Batch=2, K=8, N=6)
    EXPECT_EQ(layout_b.dims(), 3);
    ASSERT_GE(layout_b.shape().size(), 3);
    EXPECT_TRUE(symbolic::eq(layout_b.shape()[0], symbolic::integer(2))); // Batch
    EXPECT_TRUE(symbolic::eq(layout_b.shape()[1], symbolic::integer(8))); // K
    EXPECT_TRUE(symbolic::eq(layout_b.shape()[2], symbolic::integer(6))); // N

    // Verify the base matmul dimensions (last 2 dims of each tensor)
    EXPECT_TRUE(symbolic::eq(qant_matmul->m(), symbolic::integer(4))); // M from A
    EXPECT_TRUE(symbolic::eq(qant_matmul->n(), symbolic::integer(6))); // N from B
    EXPECT_TRUE(symbolic::eq(qant_matmul->k(), symbolic::integer(8))); // K (reduction dim)

    sdfg::passes::TensorToPointerConversionPass tensor_to_pointer_conversion_pass;
    tensor_to_pointer_conversion_pass.run(builder, analysis_manager);

    sdfg.validate();

    docc::qant::schedule(sdfg, "qant");

    dump_sdfg(sdfg, "batched_after_schedule");

    sdfg.validate();

    // Compile the SDFG and test execution
    std::string lib_path = docc::qant::so_compile(sdfg);

    void* h = dlopen(lib_path.c_str(), RTLD_LAZY);
    ASSERT_NE(h, nullptr) << dlerror();

    using Fn = void (*)(__bf16*, __bf16*, __bf16*);
    auto fn = reinterpret_cast<Fn>(dlsym(h, sdfg.name().c_str()));
    ASSERT_NE(fn, nullptr) << dlerror();

    const int Batch_val = 2, M_val = 4, K_val = 8, N_val = 6;
    const int total_a = Batch_val * M_val * K_val, total_b = Batch_val * K_val * N_val,
              total_c = Batch_val * M_val * N_val;
    std::vector<__bf16> A_data(total_a), B_data(total_b), C_result(total_c, (__bf16) 0.0f);
    std::vector<float> C_ref(total_c, 0.0f);

    // Initialize input data
    for (int i = 0; i < total_a; ++i) A_data[i] = (__bf16) (static_cast<float>(i % 7) * 0.5f);
    for (int i = 0; i < total_b; ++i) B_data[i] = (__bf16) (static_cast<float>(i % 5) * 0.3f);

    // Compute reference batched matmul in float
    for (int batch = 0; batch < Batch_val; ++batch) {
        for (int i = 0; i < M_val; ++i) {
            for (int j = 0; j < N_val; ++j) {
                float sum = 0.0f;
                for (int p = 0; p < K_val; ++p) {
                    sum += static_cast<float>(A_data[batch * M_val * K_val + i * K_val + p]) *
                           static_cast<float>(B_data[batch * K_val * N_val + p * N_val + j]);
                }
                C_ref[batch * M_val * N_val + i * N_val + j] = sum;
            }
        }
    }

    // Execute compiled function
    fn(A_data.data(), B_data.data(), C_result.data());

    // Verify results match reference computation
    for (int i = 0; i < total_c; ++i) {
        EXPECT_NEAR(static_cast<float>(C_result[i]), C_ref[i], 3e-2f) << "Mismatch at index " << i;
    }

    dlclose(h);
}
