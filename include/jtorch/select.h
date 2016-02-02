//
//  select.h
//
//  Created by Jonathan Tompson on 02/02/16.
//

#pragma once

#include "jcl/math/int_types.h"
#include "jcl/math/math_types.h"
#include "jtorch/tensor.h"
#include "jtorch/torch_stage.h"

namespace jtorch {

class Select : public TorchStage {
 public:
  // Constructor / Destructor
  Select(int dimension, int index);
  ~Select() override;

  TorchStageType type() const override { return SELECT_STAGE; }
  std::string name() const override { return "Select"; }
  void forwardProp(std::shared_ptr<TorchData> input) override;

  static std::unique_ptr<TorchStage> loadFromFile(std::ifstream& file);

 protected:
  int dimension_;
  int index_;

  const Tensor<float>* src_tensor_;

  // Non-copyable, non-assignable.
  Select(const Select&) = delete;
  Select& operator=(const Select&) = delete;
};

};  // namespace jtorch
