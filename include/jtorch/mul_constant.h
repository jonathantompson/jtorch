//
//  mul_constant.h
//
//  Created by Jonathan Tompson on 11/3/15.
//

#pragma once

#include "jcl/math/int_types.h"
#include "jcl/math/math_types.h"
#include "jtorch/torch_stage.h"

namespace jtorch {

class MulConstant : public TorchStage {
 public:
  // Constructor / Destructor
  MulConstant(float scalar_constant);
  ~MulConstant() override;

  TorchStageType type() const override { return MUL_CONSTANT_STAGE; }
  std::string name() const override { return "MulConstant"; }
  void forwardProp(std::shared_ptr<TorchData> input) override;

  static std::unique_ptr<TorchStage> loadFromFile(std::ifstream& file);

 protected:
  float scalar_constant_;

  void init(std::shared_ptr<TorchData> input);

  // Non-copyable, non-assignable.
  MulConstant(const MulConstant&) = delete;
  MulConstant& operator=(const MulConstant&) = delete;
};

};  // namespace jtorch
