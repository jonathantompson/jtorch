//
//  narrow.h
//
//  Created by Jonathan Tompson on 11/6/15.
//

#pragma once

#include "jcl/math/int_types.h"
#include "jcl/math/math_types.h"
#include "jtorch/tensor.h"
#include "jtorch/torch_stage.h"

namespace jtorch {

class Narrow : public TorchStage {
 public:
  // Constructor / Destructor
  Narrow(int dimension, int index, int length);
  ~Narrow() override;

  TorchStageType type() const override { return NARROW_STAGE; }
  std::string name() const override { return "Narrow"; }
  void forwardProp(std::shared_ptr<TorchData> input) override;

  static std::unique_ptr<TorchStage> loadFromFile(std::ifstream& file);

 protected:
  int dimension_;
  int index_;
  int length_;

  const Tensor<float>* src_tensor_;

  // Non-copyable, non-assignable.
  Narrow(const Narrow&) = delete;
  Narrow& operator=(const Narrow&) = delete;
};

};  // namespace jtorch
