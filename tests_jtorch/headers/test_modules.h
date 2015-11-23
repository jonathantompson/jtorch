// THE CPP FUNCTIONALITY HERE IS TO BE TESTED AGAINST "jtorch_test.lua" SCRIPT

#include <stdlib.h>
#include <cmath>
#include <cstring>
#include <thread>
#include <iostream>
#include <limits>

#include "jtorch/torch_stage.h"
#include "jtorch/jtorch.h"
#include "jtorch/tensor.h"
#include "jtorch/spatial_convolution.h"
#include "jtorch/spatial_convolution_map.h"
#include "jtorch/spatial_convolution_mm.h"
#include "jtorch/spatial_lp_pooling.h"
#include "jtorch/spatial_max_pooling.h"
#include "jtorch/spatial_subtractive_normalization.h"
#include "jtorch/spatial_divisive_normalization.h"
#include "jtorch/spatial_contrastive_normalization.h"
#include "jtorch/spatial_up_sampling_nearest.h"
#include "jtorch/spatial_batch_normalization.h"
#include "jtorch/identity.h"
#include "jtorch/linear.h"
#include "jtorch/reshape.h"
#include "jtorch/tanh.h"
#include "jtorch/threshold.h"
#include "jtorch/sequential.h"
#include "jtorch/parallel_table.h"
#include "jtorch/table.h"
#include "jtorch/join_table.h"
#include "jtorch/select_table.h"
#include "jtorch/c_add_table.h"
#include "jcl/threading/thread_pool.h"
#include "debug_util.h"
#include "file_io.h"
#include "clk/clk.h"
#include "tester.h"

TEST(Modules, Tanh) {
  Tester tester(test_path);

  jtorch::Tanh module;
  module.forwardProp(tester.data_in);
  EXPECT_TRUE(tester.testJTorchValue(module.output, "tanh_res.bin"));
}

TEST(Modules, Threshold) {
  Tester tester(test_path);

  const float threshold = 0.5f;
  const float val = 0.1f;
  jtorch::Threshold module(threshold, val);
  module.forwardProp(tester.data_in);
  EXPECT_TRUE(tester.testJTorchValue(module.output, "threshold_res.bin"));
}

TEST(Modules, Sequential) {
  Tester tester(test_path);

  std::unique_ptr<jtorch::TorchStage> model =
      jtorch::TorchStage::loadFromFile(test_path + "sequential_model.bin");
  model->forwardProp(tester.data_in);
  EXPECT_TRUE(tester.testJTorchValue(model->output, "sequential_res.bin"));
}

TEST(Modules, SpatialConvolutionMap) {
  Tester tester(test_path);

  const uint32_t num_feats_in = 5;
  const uint32_t num_feats_out = 10;
  const uint32_t fan_in = 3;
  const uint32_t filt_height = 5;
  const uint32_t filt_width = 5;
  jtorch::SpatialConvolutionMap conv(num_feats_in, num_feats_out, fan_in,
                                     filt_height, filt_width);
  for (int32_t i = 0; i < static_cast<int32_t>(num_feats_out); i++) {
    conv.biases()[i] = (float)(i + 1) / (float)num_feats_out - 0.5f;
  }
  const float sigma_x_sq = 1.0f;
  const float sigma_y_sq = 1.0f;
  for (int32_t i = 0; i < static_cast<int32_t>(num_feats_out * fan_in); i++) {
    float scale = ((float)(i + 1) / (float)(num_feats_out * fan_in));
    for (int32_t v = 0; v < static_cast<int32_t>(filt_height); v++) {
      for (int32_t u = 0; u < static_cast<int32_t>(filt_width); u++) {
        float x = (float)u - (float)(filt_width - 1) / 2.0f;
        float y = (float)v - (float)(filt_height - 1) / 2.0f;
        conv.weights()[i][v * filt_width + u] =
            scale * expf(-((x * x) / (2.0f * sigma_x_sq) +
                           (y * y) / (2.0f * sigma_y_sq)));
      }
    }
  }
  int32_t cur_filt = 0;
  for (int32_t f_out = 0; f_out < static_cast<int32_t>(num_feats_out);
       f_out++) {
    for (int32_t f_in = 0; f_in < static_cast<int32_t>(fan_in); f_in++) {
      conv.conn_table()[f_out][f_in * 2 + 1] = cur_filt;
      int32_t cur_f_in = (f_out + f_in) % num_feats_in;
      conv.conn_table()[f_out][f_in * 2] = cur_f_in;
      cur_filt++;
    }
  }
  conv.forwardProp(tester.data_in);
  tester.testJTorchValue(conv.output, "spatial_convolution_map_res.bin");
}

TEST(Modules, SpatialConvolution) {
  Tester tester(test_path);

  std::unique_ptr<jtorch::TorchStage> model = jtorch::TorchStage::loadFromFile(
      test_path + "spatial_convolution_model.bin");
  model->forwardProp(tester.data_in);
  EXPECT_TRUE(
      tester.testJTorchValue(model->output, "spatial_convolution_res.bin"));
}

TEST(Modules, SpatialConvolutionMM) {
  Tester tester(test_path);

  std::unique_ptr<jtorch::TorchStage> model = jtorch::TorchStage::loadFromFile(
      test_path + "spatial_convolution_mm_model.bin");
  model->forwardProp(tester.data_in);
  EXPECT_TRUE(
      tester.testJTorchValue(model->output, "spatial_convolution_mm_res.bin"));
}

TEST(Modules, SpatialLPPooling) {
  Tester tester(test_path);

  const float pnorm = 2;
  const int32_t pool_u = 2;
  const int32_t pool_v = 2;

  jtorch::SpatialLPPooling pool(pnorm, pool_v, pool_u);
  pool.forwardProp(tester.data_in);
  EXPECT_TRUE(
      tester.testJTorchValue(pool.output, "spatial_lp_pooling_res.bin"));
}

TEST(Modules, SpatialMaxPooling) {
  Tester tester(test_path);

  const uint32_t pool_u = 2;
  const uint32_t pool_v = 2;
  const uint32_t du = pool_u;
  const uint32_t dv = pool_v;
  const uint32_t padu = 0;
  const uint32_t padv = 0;

  jtorch::SpatialMaxPooling pool(pool_u, pool_v, du, dv, padu, padv);
  pool.forwardProp(tester.data_in);
  EXPECT_TRUE(
      tester.testJTorchValue(pool.output, "spatial_max_pooling_res.bin"));
}

TEST(Modules, SpatialMaxPoolingStride) {
  Tester tester(test_path);

  const uint32_t pool_u = 4;
  const uint32_t pool_v = 5;
  const uint32_t pool_dh = 3;
  const uint32_t pool_dw = 1;
  const uint32_t pool_padh = 0;
  const uint32_t pool_padw = 2;

  jtorch::SpatialMaxPooling pool(pool_u, pool_v, pool_dw, pool_dh, pool_padw,
                                 pool_padh);
  pool.forwardProp(tester.data_in);
  EXPECT_TRUE(tester.testJTorchValue(pool.output,
                                     "spatial_max_pooling_stride_res.bin"));
}

TEST(Modules, SpatialSubtractiveNormalization) {
  Tester tester(test_path);

  const uint32_t gauss_size = 7;
  std::shared_ptr<jtorch::Tensor<float>> kernel_1d(
      jtorch::Tensor<float>::gaussian1D(gauss_size));

  const float precision = JTORCH_FLOAT_PRECISION * 10;

  jtorch::SpatialSubtractiveNormalization sub_norm_stage(kernel_1d);
  sub_norm_stage.forwardProp(tester.data_in);
  EXPECT_TRUE(tester.testJTorchValue(
      sub_norm_stage.output, "spatial_subtractive_normalization_1d_res.bin",
      precision));

  std::shared_ptr<jtorch::Tensor<float>> kernel_2d(
      jtorch::Tensor<float>::gaussian(gauss_size));
  jtorch::SpatialSubtractiveNormalization sub_norm_stage_2d(kernel_2d);
  sub_norm_stage_2d.forwardProp(tester.data_in);
  EXPECT_TRUE(tester.testJTorchValue(
      sub_norm_stage_2d.output, "spatial_subtractive_normalization_2d_res.bin",
      precision));
}

TEST(Modules, SpatialDivisiveNormalization) {
  Tester tester(test_path);

  const uint32_t gauss_size = 7;
  std::shared_ptr<jtorch::Tensor<float>> kernel_1d(
      jtorch::Tensor<float>::gaussian1D(gauss_size));

  const float precision = JTORCH_FLOAT_PRECISION * 10;

  jtorch::SpatialDivisiveNormalization div_norm_stage(kernel_1d);
  div_norm_stage.forwardProp(tester.data_in);
  EXPECT_TRUE(tester.testJTorchValue(
      div_norm_stage.output, "spatial_divisive_normalization_1d_res.bin",
      precision));

  std::shared_ptr<jtorch::Tensor<float>> kernel_2d(
      jtorch::Tensor<float>::gaussian(gauss_size));
  jtorch::SpatialDivisiveNormalization div_norm_stage_2d(kernel_2d);
  div_norm_stage_2d.forwardProp(tester.data_in);
  EXPECT_TRUE(tester.testJTorchValue(
      div_norm_stage_2d.output, "spatial_divisive_normalization_2d_res.bin",
      precision));
}

TEST(Modules, SpatialContrastiveNormalization) {
  Tester tester(test_path);

  std::shared_ptr<jtorch::Tensor<float>> lena(
      jtorch::Tensor<float>::loadFromFile(test_path + "lena_image.bin"));

  const uint32_t kernel_size = 7;
  std::shared_ptr<jtorch::Tensor<float>> kernel2(
      new jtorch::Tensor<float>(1, &kernel_size));
  jtorch::Tensor<float>::fill(*kernel2, 1);
  jtorch::SpatialContrastiveNormalization cont_norm_stage(kernel2);
  cont_norm_stage.forwardProp(lena);
  const float precision = JTORCH_FLOAT_PRECISION * 10;
  EXPECT_TRUE(tester.testJTorchValue(
      cont_norm_stage.output, "spatial_contrastive_normalization_res.bin",
      precision));
}

TEST(Modules, Linear) {
  Tester tester(test_path);

  std::unique_ptr<jtorch::TorchStage> model =
      jtorch::TorchStage::loadFromFile(test_path + "linear_model.bin");
  model->forwardProp(tester.data_in);
  EXPECT_TRUE(tester.testJTorchValue(model->output, "linear_res.bin"));
}

TEST(Modules, Concat) {
  Tester tester(test_path);

  std::unique_ptr<jtorch::TorchStage> model =
      jtorch::TorchStage::loadFromFile(test_path + "concat_model.bin");
  model->forwardProp(tester.data_in);
  EXPECT_TRUE(tester.testJTorchValue(model->output, "concat_res.bin"));
}

TEST(Modules, ConcatTable) {
  Tester tester(test_path);

  std::unique_ptr<jtorch::TorchStage> model =
      jtorch::TorchStage::loadFromFile(test_path + "concat_table_model.bin");
  model->forwardProp(tester.data_in);
  EXPECT_TRUE(tester.testJTorchValue(model->output, "concat_table_res.bin"));
} 

TEST(Modules, Narrow) {
  Tester tester(test_path);

  std::unique_ptr<jtorch::TorchStage> model =
      jtorch::TorchStage::loadFromFile(test_path + "narrow_model.bin");
  model->forwardProp(tester.data_in);
  EXPECT_TRUE(tester.testJTorchValue(model->output, "narrow_res.bin"));
}

TEST(Modules, Identity) {
  Tester tester(test_path);

  jtorch::Identity model;
  const int32_t rand_size = 5;
  std::shared_ptr<jtorch::Tensor<float>> rand(
      jtorch::Tensor<float>::gaussian(rand_size));
  model.forwardProp(rand);
  EXPECT_EQ(model.output, rand);
}

TEST(Modules, SpatialBatchNormalization) {
  Tester tester(test_path);

  // Test Affine:
  std::unique_ptr<jtorch::TorchStage> model_affine =
      jtorch::TorchStage::loadFromFile(test_path +
                                       "batch_norm_affine_model.bin");
  std::shared_ptr<jtorch::Tensor<float>> batch_norm_in(
      jtorch::Tensor<float>::loadFromFile(test_path + "batch_norm_in.bin"));
  model_affine->forwardProp(batch_norm_in);
  EXPECT_TRUE(tester.testJTorchValue(model_affine->output,
                                     "batch_norm_affine_out.bin"));

  // Test Non-Affine:
  std::unique_ptr<jtorch::TorchStage> model =
      jtorch::TorchStage::loadFromFile(test_path + "batch_norm_model.bin");
  model->forwardProp(batch_norm_in);
  EXPECT_TRUE(tester.testJTorchValue(model->output, "batch_norm_out.bin"));
}

TEST(Modules, SpatialUpSamplingNearest) {
  Tester tester(test_path);

  const int32_t scale = 4;
  jtorch::SpatialUpSamplingNearest module(scale);
  module.forwardProp(tester.data_in);
  EXPECT_TRUE(tester.testJTorchValue(module.output,
                                     "spatial_up_sampling_nearest_res.bin"));
}

TEST(Modules, SelectTable) {
  Tester tester(test_path);

  const int32_t table_size = 5;
  std::shared_ptr<jtorch::TorchData> input(new jtorch::Table());
  jtorch::Table* table_input = (jtorch::Table*)input.get();
  for (int32_t i = 0; i < table_size; i++) {
    table_input->add(std::shared_ptr<jtorch::Tensor<float>>(
        jtorch::Tensor<float>::gaussian1D(i + 1)));
  }

  for (int32_t i = 0; i < table_size; i++) {
    std::unique_ptr<jtorch::SelectTable> module(new jtorch::SelectTable(i));
    module->forwardProp(input);
    EXPECT_EQ(module->output.get(), (*table_input)(i).get());
    EXPECT_EQ(TO_TENSOR_PTR(module->output.get())->nelems(), (uint32_t)(i + 1));
  }
}

TEST(Modules, CAddTable) {
  Tester tester(test_path);

  const int32_t table_size = 5;
  const int32_t tensor_size = 5;
  std::shared_ptr<jtorch::TorchData> input(new jtorch::Table());
  jtorch::Table* table_input = (jtorch::Table*)input.get();
  for (int32_t i = 0; i < table_size; i++) {
    table_input->add(std::shared_ptr<jtorch::Tensor<float>>(
        jtorch::Tensor<float>::gaussian(tensor_size)));
    jtorch::Tensor<float>::mul(*TO_TENSOR_PTR((*table_input)(i).get()),
                               (float)(i + 1));
  }

  // Add the tensors to get the ground truth
  std::unique_ptr<float[]> gt(new float[tensor_size * tensor_size]);
  std::unique_ptr<float[]> temp(new float[tensor_size * tensor_size]);
  memset(gt.get(), 0, sizeof(gt[0]) * tensor_size * tensor_size);
  for (int32_t i = 0; i < table_size; i++) {
    TO_TENSOR_PTR((*table_input)(i).get())->getData(temp.get());
    for (uint32_t i = 0; i < tensor_size * tensor_size; i++) {
      gt[i] += temp[i];
    }
  }

  std::unique_ptr<jtorch::CAddTable> module(new jtorch::CAddTable());
  module->forwardProp(input);
  TO_TENSOR_PTR(module->output.get())->getData(temp.get());

  for (int32_t i = 0; i < tensor_size * tensor_size; i++) {
    EXPECT_TRUE(fabsf(temp[i] - gt[i]) < JTORCH_FLOAT_PRECISION);
  }
}

TEST(Modules, CompoundModel) {
  Tester tester(test_path);

  std::unique_ptr<jtorch::TorchStage> model =
      jtorch::TorchStage::loadFromFile(test_path + "test_model.bin");
  model->forwardProp(tester.data_in);

  // Some debugging if things go wrong:
  EXPECT_EQ(model->type(), jtorch::SEQUENTIAL_STAGE);
  jtorch::Sequential* seq = (jtorch::Sequential*)model.get();
  const jtorch::TorchStageType stages[] = {jtorch::SPATIAL_CONVOLUTION_STAGE,
                                           jtorch::TANH_STAGE,
                                           jtorch::THRESHOLD_STAGE,
                                           jtorch::SPATIAL_MAX_POOLING_STAGE,
                                           jtorch::SPATIAL_CONVOLUTION_MM_STAGE,
                                           jtorch::RESHAPE_STAGE,
                                           jtorch::LINEAR_STAGE};

  EXPECT_EQ(seq->size(), sizeof(stages) / sizeof(stages[0]));

  for (uint32_t i = 0; i < sizeof(stages) / sizeof(stages[0]); i++) {
    jtorch::TorchStage* stage = seq->get(i);
    static_cast<void>(stage);
    EXPECT_EQ(stage->type(), stages[i]);
  }

  EXPECT_TRUE(tester.testJTorchValue(model->output, "test_model_res.bin"));
}

TEST(Modules, ProfileConvolution) {
  const uint32_t fin = 128, fout = 512, kw = 11, kh = 11, pad = 5, imw = 90,
                 imh = 60;
  const double t_test = 1.0;
  double t_start, t_end;
  uint64_t niters;
  jtorch::SpatialConvolution conv(fin, fout, kh, kw, pad);
  jtorch::SpatialConvolutionMM conv_mm(fin, fout, kh, kw, pad, pad);
  const uint32_t size[3] = {imw, imh, fin};
  std::shared_ptr<jtorch::Tensor<float>> input(
      new jtorch::Tensor<float>(3, size));
  clk::Clk clk;

  jtorch::Tensor<float>::fill(*conv.weights(), 1);
  jtorch::Tensor<float>::fill(*conv.biases(), 1);
  jtorch::Tensor<float>::fill(*input, 1);

  std::cout << std::endl;
  std::cout << "\tProfiling SpatialConvolutionMM for " << t_test << " seconds "
            << std::endl;
  t_start = clk.getTime();
  t_end = t_start;
  niters = 0;
  while (t_end - t_start < t_test) {
    conv_mm.forwardProp(input);
    niters++;
    jtorch::Sync();
    t_end = clk.getTime();
  }
  std::cout << "\t\tExecution time: " << (t_end - t_start) / (double)niters
            << " seconds per FPROP" << std::endl;

  std::cout << "\tProfiling SpatialConvolution for " << t_test << " seconds"
            << std::endl;
  t_start = clk.getTime();
  t_end = t_start;
  niters = 0;
  while (t_end - t_start < t_test) {
    conv.forwardProp(input);
    niters++;
    jtorch::Sync();
    t_end = clk.getTime();
  }
  std::cout << "\t\tExecution time: " << (t_end - t_start) / (double)niters
            << " seconds per FPROP" << std::endl;
}
