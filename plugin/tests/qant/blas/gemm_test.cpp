#include <gtest/gtest.h>

#include <dlfcn.h>
#include <stdfloat>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/passes/pipeline.h"

#include "docc/qant/passes/expansion_pass.h"
#include "docc/qant/plugin.h"
#include "docc/qant/qant.h"

using namespace sdfg;

TEST(GEMMTest, GEMM_2D_BFloat16) {
    // Test GEMM: C = alpha * A @ B + beta * C with BFloat16
    // This mirrors the Python test_matmul_bf16: a @ b with 2D arrays
    // Python handle_gemm creates: alpha=1.0, beta=0.0, trans_a=No, trans_b=No
    builder::StructuredSDFGBuilder builder("gemm_bf16_2d", FunctionType_CPU);

    auto& sdfg = builder.subject();

    const int M = 4, K = 8, N = 6;

    types::Scalar desc(types::PrimitiveType::BFloat);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr, true);
    builder.add_container("b", desc_ptr, true);
    builder.add_container("c", desc_ptr, true);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& c_input_node = builder.add_access(block, "c");
    auto& c_output_node = builder.add_access(block, "c");

    auto& alpha_node = builder.add_constant(block, "1.0", desc);
    auto& beta_node = builder.add_constant(block, "0.0", desc);

    auto& gemm_node = static_cast<math::blas::GEMMNode&>(builder.add_library_node<math::blas::GEMMNode>(
        block,
        DebugInfo(),
        math::blas::ImplementationType_BLAS,
        math::blas::BLAS_Precision::h,
        math::blas::BLAS_Layout::RowMajor,
        math::blas::BLAS_Transpose::No,
        math::blas::BLAS_Transpose::No,
        symbolic::integer(M),
        symbolic::integer(N),
        symbolic::integer(K),
        symbolic::integer(K), // lda = K (no trans)
        symbolic::integer(N), // ldb = N (no trans)
        symbolic::integer(N) // ldc = N
    ));

    builder.add_computational_memlet(block, a_node, gemm_node, "__A", {symbolic::zero()}, desc_ptr, block.debug_info());
    builder.add_computational_memlet(block, b_node, gemm_node, "__B", {symbolic::zero()}, desc_ptr, block.debug_info());
    builder
        .add_computational_memlet(block, c_input_node, gemm_node, "__C", {symbolic::zero()}, desc_ptr, block.debug_info());
    builder.add_computational_memlet(block, alpha_node, gemm_node, "__alpha", {}, desc, block.debug_info());
    builder.add_computational_memlet(block, beta_node, gemm_node, "__beta", {}, desc, block.debug_info());
    builder.add_computational_memlet(
        block, gemm_node, "__C", c_output_node, {symbolic::zero()}, desc_ptr, block.debug_info()
    );

    sdfg.validate();

    analysis::AnalysisManager analysis_manager(sdfg);

    // Run QantRemappingPass (sets implementation type to QANT for GEMM)
    sdfg::passes::Pipeline expansion("QantRemapping");
    expansion.register_pass<sdfg::passes::QantRemappingPass>();
    expansion.run(builder, analysis_manager);

    sdfg.validate();

    // Verify the GEMM node now has QANT implementation type
    auto library_nodes = block.dataflow().library_nodes();
    EXPECT_EQ(library_nodes.size(), 1);
    auto* qant_gemm = dynamic_cast<sdfg::math::blas::GEMMNode*>(*library_nodes.begin());
    EXPECT_TRUE(qant_gemm);
    EXPECT_EQ(qant_gemm->implementation_type(), docc::qant::ImplementationType_QANT.value());

    docc::qant::schedule(sdfg, "qant");

    sdfg.validate();

    EXPECT_NO_THROW(std::string lib_path = docc::qant::so_compile(sdfg);
}
