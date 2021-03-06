//
//  tanh.h
//
//  Created by Jonathan Tompson on 4/1/13.
//

#pragma once

#include "jcl/math/int_types.h"
#include "jcl/math/math_types.h"
#include "jtorch/torch_stage.h"

namespace jtorch {

class Tanh : public TorchStage {
 public:
  // Constructor / Destructor
  Tanh();
  ~Tanh() override;

  TorchStageType type() const override { return TANH_STAGE; }
  std::string name() const override { return "Tanh"; }
  void forwardProp(std::shared_ptr<TorchData> input) override;

  static std::unique_ptr<TorchStage> loadFromFile(std::ifstream& file);

 protected:
  void init(std::shared_ptr<TorchData> input);

  // Non-copyable, non-assignable.
  Tanh(const Tanh&) = delete;
  Tanh& operator=(const Tanh&) = delete;
};

};  // namespace jtorch
