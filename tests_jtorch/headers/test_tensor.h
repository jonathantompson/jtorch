// THE CPP FUNCTIONALITY HERE IS TO BE TESTED AGAINST "jtorch_test.lua" SCRIPT

#define _USE_MATH_DEFINES
#include <math.h>

#include "tester.h"

#define JTORCH_TENSOR_PRECISION 1e-6f

TEST(Tensor, SaveAndLoad) {
  Tester tester(test_path);

  const uint32_t idim = 3;
  const uint32_t num_feats_in = 5;
  const uint32_t width = 10;
  const uint32_t height = 10;
  const uint32_t isize[idim] = {width, height, num_feats_in};
  std::unique_ptr<float[]> data_in_cpu(
      new float[width * height * num_feats_in]);

  // First create the input.
  for (uint32_t f = 0; f < num_feats_in; f++) {
    float val = (f + 1) - (float)(width * height) / 16.0f;
    for (uint32_t v = 0; v < height; v++) {
      for (uint32_t u = 0; u < width; u++) {
        data_in_cpu[f * width * height + v * width + u] = val;
        val += 1.0f / 8.0f;
      }
    }
  }

  // Now load a tensor created in torch (that was created using the same
  // procedure) and check it for correctness.
  std::shared_ptr<jtorch::Tensor<float>> data_in_torch(
      jtorch::Tensor<float>::loadFromFile(test_path + "data_in.bin"));
  std::unique_ptr<float[]> data_in_torch_cpu(
      new float[width * height * num_feats_in]);
  data_in_torch->getData(data_in_torch_cpu.get());

  for (uint32_t i = 0; i < width * height * num_feats_in; i++) {
    EXPECT_APPROX_EQ(data_in_torch_cpu[i], data_in_cpu[i],
                     JTORCH_TENSOR_PRECISION)
  }

  // Test the tester's version of the same test.
  EXPECT_TRUE(tester.testJTorchValue(data_in_torch, "data_in.bin"));

  // Save the tensor to disk.
  jtorch::Tensor<float>::saveToFile(*data_in_torch,
                                    test_path + "data_in_cpp.bin");

  // Load that same tensor back and make sure it is OK.
  std::shared_ptr<jtorch::Tensor<float>> data_in_load(
      jtorch::Tensor<float>::loadFromFile(test_path + "data_in_cpp.bin"));
  EXPECT_TRUE(tester.testJTorchValue(data_in_load, "data_in.bin"));
}

TEST(Tensor, RandSumMaxMin) {
  const uint32_t dim = 4;
  const uint32_t size[dim] = {101, 11, 12, 2};
  std::shared_ptr<jtorch::Tensor<float>> rand =
      jtorch::Tensor<float>::slowRand(dim, size);

  // This isn't a great test, but make sure the average is about 0.5.
  std::unique_ptr<float[]> rand_cpu(new float[rand->nelems()]);
  rand->getData(rand_cpu.get());

  float min = std::numeric_limits<float>::infinity();
  float max = -std::numeric_limits<float>::infinity();
  float sum = 0;
  for (uint32_t i = 0; i < rand->nelems(); i++) {
    sum += rand_cpu[i];
    min = std::min<float>(min, rand_cpu[i]);
    max = std::max<float>(max, rand_cpu[i]);
  }
  EXPECT_LE(max, 1);
  EXPECT_GE(min, 0);
  float mean = sum / rand->nelems();
  EXPECT_LT(fabsf(mean - 0.5f), 10.0f * JTORCH_FLOAT_PRECISION);

  EXPECT_LT(fabsf(sum - jtorch::Tensor<float>::slowSum(*rand)),
            JTORCH_FLOAT_PRECISION);
  EXPECT_EQ(max, jtorch::Tensor<float>::slowMax(*rand));
  EXPECT_EQ(min, jtorch::Tensor<float>::slowMin(*rand));
  EXPECT_LT(fabsf(mean - jtorch::Tensor<float>::slowMean(*rand)),
            JTORCH_FLOAT_PRECISION);
}

TEST(Tensor, CopyResizeAs) {
  const uint32_t dim = 4;
  const uint32_t size[dim] = {101, 11, 12, 2};
  std::shared_ptr<jtorch::Tensor<float>> rand =
      jtorch::Tensor<float>::slowRand(dim, size);
  const uint32_t copy_dim = 2;
  const uint32_t copy_size[dim] = {101, 11};
  std::shared_ptr<jtorch::Tensor<float>> rand_copy(
      new jtorch::Tensor<float>(copy_dim, copy_size));

  // The resizeAs call will force a resize of the underlining storage.
  EXPECT_TRUE(rand_copy->resizeAs(*rand));
  EXPECT_TRUE(rand_copy->isSameSizeAs(*rand));

  // Another resizeAs call shouldn't allocate new memory.
  EXPECT_FALSE(rand_copy->resizeAs(*rand));

  jtorch::Tensor<float>::copy(*rand_copy, *rand);

  // This isn't a great test, but make sure the average is about 0.5.
  std::unique_ptr<float[]> rand_cpu(new float[rand->nelems()]);
  rand->getData(rand_cpu.get());
  std::unique_ptr<float[]> rand_copy_cpu(new float[rand_copy->nelems()]);
  rand_copy->getData(rand_copy_cpu.get());

  for (uint32_t i = 0; i < rand->nelems(); i++) {
    EXPECT_EQ(rand_copy_cpu[i], rand_cpu[i]);
  }

  // Shrinking the tensor shouldn't allocate new memory.
  EXPECT_FALSE(rand_copy->resize(copy_dim, copy_size));
}

TEST(Tensor, Add) {
  const uint32_t dim = 4;
  const uint32_t size[dim] = {2, 3, 5, 7};

  std::shared_ptr<jtorch::Tensor<float>> a =
      jtorch::Tensor<float>::slowRand(dim, size);
  std::shared_ptr<jtorch::Tensor<float>> b =
      jtorch::Tensor<float>::slowRand(dim, size);
  std::shared_ptr<jtorch::Tensor<float>> c(
      new jtorch::Tensor<float>(dim, size));

  jtorch::Tensor<float>::add(*c, *a, *b);  // c = a + b

  EXPECT_TRUE(a->isSameSizeAs(*b));
  EXPECT_TRUE(a->isSameSizeAs(*c));

  const uint32_t nelems = a->nelems();
  std::unique_ptr<float[]> a_cpu(new float[nelems]);
  a->getData(a_cpu.get());
  std::unique_ptr<float[]> b_cpu(new float[nelems]);
  b->getData(b_cpu.get());
  std::unique_ptr<float[]> c_cpu(new float[nelems]);
  c->getData(c_cpu.get());

  for (uint32_t i = 0; i < nelems; i++) {
    EXPECT_APPROX_EQ(c_cpu[i], a_cpu[i] + b_cpu[i], JTORCH_TENSOR_PRECISION);
  }

  // Also try adding a scalar:
  jtorch::Tensor<float>::copy(*c, *a);                       // c = a
  jtorch::Tensor<float>::add(*c, static_cast<float>(M_PI));  // c = a + pi
  c->getData(c_cpu.get());
  for (uint32_t i = 0; i < nelems; i++) {
    EXPECT_APPROX_EQ(c_cpu[i], a_cpu[i] + static_cast<float>(M_PI),
                     JTORCH_TENSOR_PRECISION);
  }
}

TEST(Tensor, Sub) {
  const uint32_t dim = 4;
  const uint32_t size[dim] = {2, 3, 5, 7};

  std::shared_ptr<jtorch::Tensor<float>> a =
      jtorch::Tensor<float>::slowRand(dim, size);
  std::shared_ptr<jtorch::Tensor<float>> b =
      jtorch::Tensor<float>::slowRand(dim, size);
  std::shared_ptr<jtorch::Tensor<float>> c(
      new jtorch::Tensor<float>(dim, size));

  jtorch::Tensor<float>::sub(*c, *a, *b);  // c = a - b

  EXPECT_TRUE(a->isSameSizeAs(*b));
  EXPECT_TRUE(a->isSameSizeAs(*c));

  const uint32_t nelems = a->nelems();
  std::unique_ptr<float[]> a_cpu(new float[nelems]);
  a->getData(a_cpu.get());
  std::unique_ptr<float[]> b_cpu(new float[nelems]);
  b->getData(b_cpu.get());
  std::unique_ptr<float[]> c_cpu(new float[nelems]);
  c->getData(c_cpu.get());

  for (uint32_t i = 0; i < nelems; i++) {
    EXPECT_APPROX_EQ(c_cpu[i], a_cpu[i] - b_cpu[i], JTORCH_TENSOR_PRECISION);
  }
}

TEST(Tensor, Clone) {
  const uint32_t dim = 4;
  const uint32_t size[dim] = {2, 3, 5, 7};

  std::shared_ptr<jtorch::Tensor<float>> a =
      jtorch::Tensor<float>::slowRand(dim, size);
  std::shared_ptr<jtorch::Tensor<float>> a_copy =
      jtorch::Tensor<float>::clone(*a);

  EXPECT_TRUE(a->isSameSizeAs(*a_copy));

  const uint32_t nelems = a->nelems();
  std::unique_ptr<float[]> a_cpu(new float[nelems]);
  a->getData(a_cpu.get());
  std::unique_ptr<float[]> a_copy_cpu(new float[nelems]);
  a_copy->getData(a_copy_cpu.get());

  for (uint32_t i = 0; i < nelems; i++) {
    EXPECT_EQ(a_cpu[i], a_copy_cpu[i]);
  }
}

TEST(Tensor, Mul) {
  const uint32_t dim = 4;
  const uint32_t size[dim] = {2, 3, 5, 7};

  std::shared_ptr<jtorch::Tensor<float>> a =
      jtorch::Tensor<float>::slowRand(dim, size);
  std::shared_ptr<jtorch::Tensor<float>> a_copy =
      jtorch::Tensor<float>::clone(*a);

  jtorch::Tensor<float>::mul(*a, 0.5f);  // a = a * 0.5

  EXPECT_TRUE(a->isSameSizeAs(*a_copy));

  const uint32_t nelems = a->nelems();
  std::unique_ptr<float[]> a_cpu(new float[nelems]);
  a->getData(a_cpu.get());
  std::unique_ptr<float[]> a_copy_cpu(new float[nelems]);
  a_copy->getData(a_copy_cpu.get());

  for (uint32_t i = 0; i < nelems; i++) {
    EXPECT_APPROX_EQ(a_cpu[i], a_copy_cpu[i] * 0.5f, JTORCH_TENSOR_PRECISION);
  }
}

TEST(Tensor, SelectOuter) {
  const uint32_t dim = 4;
  const uint32_t size[dim] = {2, 3, 4, 5};

  std::shared_ptr<jtorch::Tensor<float>> a =
      jtorch::Tensor<float>::slowRand(dim, size);
  std::shared_ptr<jtorch::Tensor<float>> a_copy =
      jtorch::Tensor<float>::clone(*a);

  // We're going to calculate a_f = a_f * (1/f) + f
  // Note: mul and add have already been tested.
  const uint32_t nfeats = size[dim - 1];
  for (uint32_t f = 0; f < nfeats; f++) {
    std::shared_ptr<jtorch::Tensor<float>> a_f =
        jtorch::Tensor<float>::selectOuterDim(*a, f);
    jtorch::Tensor<float>::mul(*a_f, 1.0f / static_cast<float>(f));
    jtorch::Tensor<float>::add(*a_f, static_cast<float>(f));
  }

  EXPECT_TRUE(a->isSameSizeAs(*a_copy));

  const uint32_t nelems = a->nelems();
  std::unique_ptr<float[]> a_cpu(new float[nelems]);
  a->getData(a_cpu.get());
  std::unique_ptr<float[]> a_copy_cpu(new float[nelems]);
  a_copy->getData(a_copy_cpu.get());

  const uint32_t im_size = a->nelems() / nfeats;
  for (uint32_t f = 0; f < nfeats; f++) {
    for (uint32_t i = 0; i < im_size; i++) {
      const uint32_t index = f * im_size + i;
      EXPECT_APPROX_EQ(
          a_copy_cpu[index] * 1.0f / static_cast<float>(f) + static_cast<float>(f),
          a_cpu[index], JTORCH_TENSOR_PRECISION);
    }
  }
}

TEST(Tensor, NarrowOuter) {
  const uint32_t dim = 4;
  const uint32_t size[dim] = {2, 3, 4, 5};

  std::shared_ptr<jtorch::Tensor<float>> a =
      jtorch::Tensor<float>::slowRand(dim, size);
  std::shared_ptr<jtorch::Tensor<float>> a_copy =
      jtorch::Tensor<float>::clone(*a);

  // We're going to multiply outer dimension indices 1, 2 and 3 by 0.5.
  // Note: mul has already been tested.
  const uint32_t i_start = 1;
  const uint32_t len = 3;
  std::shared_ptr<jtorch::Tensor<float>> a_12_slice =
      jtorch::Tensor<float>::narrowOuterDim(*a, i_start, len);

  jtorch::Tensor<float>::mul(*a_12_slice, 0.5f);

  EXPECT_TRUE(a->isSameSizeAs(*a_copy));

  const uint32_t nelems = a->nelems();
  std::unique_ptr<float[]> a_cpu(new float[nelems]);
  a->getData(a_cpu.get());
  std::unique_ptr<float[]> a_copy_cpu(new float[nelems]);
  a_copy->getData(a_copy_cpu.get());

  const uint32_t nfeats = size[dim - 1];
  const uint32_t im_size = a->nelems() / nfeats;
  for (uint32_t f = 0; f < nfeats; f++) {
    for (uint32_t i = 0; i < im_size; i++) {
      const uint32_t index = f * im_size + i;
      if (i_start <= f && f < (i_start + len)) {
        EXPECT_APPROX_EQ(a_copy_cpu[index] * 0.5f, a_cpu[index],
                         JTORCH_TENSOR_PRECISION);
      } else {
        EXPECT_EQ(a_copy_cpu[index], a_cpu[index]);
      }
    }
  }
}
