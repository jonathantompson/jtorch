#include "tester.h"

#include "jtorch/tensor.h"

Tester::Tester(const std::string& test_path) {
  test_path_ = test_path;
  data_in = jtorch::Tensor<float>::loadFromFile(test_path_ + "data_in.bin");
}

Tester::~Tester() {}

// Note: this function is tested for correctness in test_tensor.h.
bool Tester::testJTorchValue(std::shared_ptr<jtorch::TorchData> torch_data,
                             const std::string& filename,
                             const float precision) {
  jtorch::Tensor<float>* data = TO_TENSOR_PTR(torch_data.get());
  std::unique_ptr<float[]> correct_data(new float[data->nelems()]);
  std::unique_ptr<float[]> model_data(new float[data->nelems()]);
  memset(model_data.get(), 0, sizeof(model_data[0]) * data->nelems());
  memset(correct_data.get(), 0, sizeof(correct_data[0]) * data->nelems());

  std::shared_ptr<jtorch::Tensor<float>> correct_data_tensor(
      jtorch::Tensor<float>::loadFromFile(test_path_ + filename));

  bool data_correct = true;
  if (!correct_data_tensor->isSameSizeAs(*data)) {
    std::cout << "Test FAILED (size mismatch)!: " << filename << std::endl;
    data_correct = false;
  } else {
    correct_data_tensor->getData(correct_data.get());
    data->getData(model_data.get());
    for (uint32_t i = 0; i < data->nelems() && data_correct; i++) {
      float delta;
      if (fabsf(correct_data[i]) > precision) {
        delta = fabsf(model_data[i] - correct_data[i]) /
                std::max<float>(fabsf(correct_data[i]), kLooseEpsilon);
      } else {
        delta = fabsf(model_data[i] - correct_data[i]);
      }
      if (delta > precision) {
        data_correct = false;
        std::cout << std::endl;
        for (uint32_t repeat = 0; repeat < 5; repeat++) {
          for (uint32_t cnt = 0; cnt < 60; cnt++) {
            std::cout << "*";
          }
          std::cout << std::endl;
        }
        std::cout << "index " << i << " incorrect!: " << std::endl;
        std::cout << std::fixed << std::setprecision(15);
        std::cout << "  model_data[" << i << "] = " << model_data[i]
                  << std::endl;
        std::cout << "  correct_data[" << i << "] = " << correct_data[i]
                  << std::endl;
        std::cout << "  normalized delta: " << delta << std::endl;
        for (uint32_t repeat = 0; repeat < 5; repeat++) {
          for (uint32_t cnt = 0; cnt < 60; cnt++) {
            std::cout << "*";
          }
          std::cout << std::endl;
        }
      }
    }
  }
  return data_correct;
}
