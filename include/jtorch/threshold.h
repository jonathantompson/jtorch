//
//  threshold.h
//
//  Created by Jonathan Tompson on 4/1/13.
//

#pragma once

#include "jcl/math/int_types.h"
#include "jcl/math/math_types.h"
#include "jtorch/torch_stage.h"

namespace jtorch {

class Threshold : public TorchStage {
 public:
  // Constructor / Destructor
  Threshold();  // Use default threshold and val
  Threshold(const float threshold, const float val);
  ~Threshold() override;

  TorchStageType type() const override { return THRESHOLD_STAGE; }
  std::string name() const override { return "Threshold"; }
  void forwardProp(std::shared_ptr<TorchData> input) override;

  static std::unique_ptr<TorchStage> loadFromFile(std::ifstream& file);

 protected:
  float threshold_;  // Single threshold value
  float val_;  // Single output value (when input < threshold)

  void init(std::shared_ptr<TorchData> input);

  // Non-copyable, non-assignable.
  Threshold(const Threshold&) = delete;
  Threshold& operator=(const Threshold&) = delete;
};

};  // namespace jtorch
