#include <gtest/gtest.h>

#include <cmath>
#include <dlfcn.h>
#include <vector>

#include <sdfg/passes/dataflow/tensor_to_pointer_conversion.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/batchnorm_node.h"
#include "sdfg/passes/pipeline.h"
#include "sdfg_debug_dump.h"

#include "docc/qant/passes/remapping_pass.h"
#include "docc/qant/plugin.h"
#include "docc/qant/qant.h"

using namespace sdfg;


TEST(BatchNormTest, BatchNorm2D_QANT_Simple) {
    // Test simple 2D batch normalization via QANT: Batch[1, 2, 4, 4] with 2 channels
    builder::StructuredSDFGBuilder builder("batchnorm2d_qant_simple", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("Batch", desc_ptr, true);
    builder.add_container("Var", desc_ptr, true);
    builder.add_container("E", desc_ptr, true);
    builder.add_container("Gamma", desc_ptr, true);
    builder.add_container("Beta", desc_ptr, true);
    builder.add_container("B_out", desc_ptr, true);

    auto& block = builder.add_block(sdfg.root());

    auto& batch_node = builder.add_access(block, "Batch");
    auto& var_node = builder.add_access(block, "Var");
    auto& e_node = builder.add_access(block, "E");
    auto& gamma_node = builder.add_access(block, "Gamma");
    auto& beta_node = builder.add_access(block, "Beta");
    auto& b_out_node = builder.add_access(block, "B_out");
    auto& epsilon_node = builder.add_constant(block, "0.00001", desc);

    const int N = 1, C = 2, H = 4, W = 4;

    symbolic::MultiExpression batch_shape = {
        symbolic::integer(N), symbolic::integer(C), symbolic::integer(H), symbolic::integer(W)
    };
    symbolic::MultiExpression feature_shape = {symbolic::integer(C)};

    types::Tensor batch_tensor(desc.primitive_type(), batch_shape);
    types::Tensor var_tensor(desc.primitive_type(), feature_shape);
    types::Tensor e_tensor(desc.primitive_type(), feature_shape);
    types::Tensor gamma_tensor(desc.primitive_type(), feature_shape);
    types::Tensor beta_tensor(desc.primitive_type(), feature_shape);
    types::Tensor b_out_tensor(desc.primitive_type(), batch_shape);

    auto& batchnorm_node =
        dynamic_cast<math::tensor::BatchNormNode&>(builder.add_library_node<math::tensor::BatchNormNode>(
            block, DebugInfo(), math::tensor::TensorLayout(batch_shape), types::Float
        ));

    builder.add_computational_memlet(block, batch_node, batchnorm_node, "Batch", {}, batch_tensor, block.debug_info());
    builder.add_computational_memlet(block, var_node, batchnorm_node, "Var", {}, var_tensor, block.debug_info());
    builder.add_computational_memlet(block, e_node, batchnorm_node, "E", {}, e_tensor, block.debug_info());
    builder.add_computational_memlet(block, gamma_node, batchnorm_node, "Gamma", {}, gamma_tensor, block.debug_info());
    builder.add_computational_memlet(block, beta_node, batchnorm_node, "Beta", {}, beta_tensor, block.debug_info());
    builder.add_computational_memlet(block, epsilon_node, batchnorm_node, "epsilon", {}, desc, block.debug_info());
    builder.add_computational_memlet(block, b_out_node, batchnorm_node, "B_out", {}, b_out_tensor, block.debug_info());

    sdfg.validate();

    analysis::AnalysisManager analysis_manager(sdfg);

    // Run remapping pass: BatchNormNode → QantBatchNormNode
    sdfg::passes::Pipeline remapping("QantRemapping");
    remapping.register_pass<sdfg::passes::QantRemappingPass>();
    remapping.run(builder, analysis_manager);

    sdfg.validate();

    auto library_nodes = block.dataflow().library_nodes();
    EXPECT_EQ(library_nodes.size(), 1);
    auto* new_node = dynamic_cast<sdfg::math::tensor::BatchNormNode*>(*library_nodes.begin());
    EXPECT_TRUE(new_node);
    EXPECT_EQ(new_node->implementation_type(), docc::qant::ImplementationType_QANT);

    sdfg::passes::TensorToPointerConversionPass tensor_to_pointer_conversion_pass;
    tensor_to_pointer_conversion_pass.run(builder, analysis_manager);

    sdfg.validate();

    docc::qant::schedule(sdfg, "qant");

    sdfg.validate();

    std::string lib_path = docc::qant::so_compile(sdfg);
}


TEST(BatchNormTest, BatchNorm2D_QANT_Batched) {
    // Test batched 2D batch normalization via QANT: Batch[4, 2, 3, 3] with 2 channels, 4 batches
    builder::StructuredSDFGBuilder builder("batchnorm2d_qant_batched", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("Batch", desc_ptr, true);
    builder.add_container("Var", desc_ptr, true);
    builder.add_container("E", desc_ptr, true);
    builder.add_container("Gamma", desc_ptr, true);
    builder.add_container("Beta", desc_ptr, true);
    builder.add_container("B_out", desc_ptr, true);

    auto& block = builder.add_block(sdfg.root());

    auto& batch_node = builder.add_access(block, "Batch");
    auto& var_node = builder.add_access(block, "Var");
    auto& e_node = builder.add_access(block, "E");
    auto& gamma_node = builder.add_access(block, "Gamma");
    auto& beta_node = builder.add_access(block, "Beta");
    auto& b_out_node = builder.add_access(block, "B_out");
    auto& epsilon_node = builder.add_constant(block, "0.00001", desc);

    const int N = 4, C = 2, H = 3, W = 3;

    symbolic::MultiExpression batch_shape = {
        symbolic::integer(N), symbolic::integer(C), symbolic::integer(H), symbolic::integer(W)
    };
    symbolic::MultiExpression feature_shape = {symbolic::integer(C)};

    types::Tensor batch_tensor(desc.primitive_type(), batch_shape);
    types::Tensor var_tensor(desc.primitive_type(), feature_shape);
    types::Tensor e_tensor(desc.primitive_type(), feature_shape);
    types::Tensor gamma_tensor(desc.primitive_type(), feature_shape);
    types::Tensor beta_tensor(desc.primitive_type(), feature_shape);
    types::Tensor b_out_tensor(desc.primitive_type(), batch_shape);

    auto& batchnorm_node =
        dynamic_cast<math::tensor::BatchNormNode&>(builder.add_library_node<math::tensor::BatchNormNode>(
            block, DebugInfo(), math::tensor::TensorLayout(batch_shape), types::Float
        ));

    builder.add_computational_memlet(block, batch_node, batchnorm_node, "Batch", {}, batch_tensor, block.debug_info());
    builder.add_computational_memlet(block, var_node, batchnorm_node, "Var", {}, var_tensor, block.debug_info());
    builder.add_computational_memlet(block, e_node, batchnorm_node, "E", {}, e_tensor, block.debug_info());
    builder.add_computational_memlet(block, gamma_node, batchnorm_node, "Gamma", {}, gamma_tensor, block.debug_info());
    builder.add_computational_memlet(block, beta_node, batchnorm_node, "Beta", {}, beta_tensor, block.debug_info());
    builder.add_computational_memlet(block, epsilon_node, batchnorm_node, "epsilon", {}, desc, block.debug_info());
    builder.add_computational_memlet(block, b_out_node, batchnorm_node, "B_out", {}, b_out_tensor, block.debug_info());

    sdfg.validate();

    analysis::AnalysisManager analysis_manager(sdfg);

    // Run remapping pass
    sdfg::passes::Pipeline remapping("QantRemapping");
    remapping.register_pass<sdfg::passes::QantRemappingPass>();
    remapping.run(builder, analysis_manager);

    sdfg.validate();

    auto library_nodes = block.dataflow().library_nodes();
    EXPECT_EQ(library_nodes.size(), 1);
    auto* new_node = dynamic_cast<sdfg::math::tensor::BatchNormNode*>(*library_nodes.begin());
    EXPECT_TRUE(new_node);
    EXPECT_EQ(new_node->implementation_type(), docc::qant::ImplementationType_QANT);

    sdfg::passes::TensorToPointerConversionPass tensor_to_pointer_conversion_pass;
    tensor_to_pointer_conversion_pass.run(builder, analysis_manager);

    sdfg.validate();

    docc::qant::schedule(sdfg, "qant");

    sdfg.validate();

    std::string lib_path = docc::qant::so_compile(sdfg);
}
