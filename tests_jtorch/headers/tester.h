//
//  tester.h
//
//  Created by Jonathan Tompson on 11/05/15.
//
//  A class to check tensor values against saved tensors on disk. I also used
//  to create and destroy jtorch contexts here, but it makes the tests very
//  slow.

#pragma once

#include <memory>
#include <string>

#include "jtorch/tensor.h"
#include "jtorch/torch_data.h"

#define JTORCH_FLOAT_PRECISION 1e-4f

// Tester will create the jtorch module.
struct Tester {
 public:
  Tester(const std::string& test_path);
  ~Tester();
  std::shared_ptr<jtorch::Tensor<float>> data_in;
  bool testJTorchValue(std::shared_ptr<jtorch::TorchData> torch_data,
                       const std::string& filename,
                       const float precision = JTORCH_FLOAT_PRECISION);
 private:
  std::string test_path_;
};
