//
//  threshold.h
//
//  Created by Jonathan Tompson on 4/1/13.
//

#pragma once

#include <mutex>
#include <condition_variable>
#include "jcl/math/int_types.h"
#include "jcl/math/math_types.h"
#include "jtorch/torch_stage.h"

namespace jcl {
namespace data_str {
template <typename T>
class VectorManaged;
}
}

namespace jtorch {

class Threshold : public TorchStage {
 public:
  // Constructor / Destructor
  Threshold();
  ~Threshold() override;

  TorchStageType type() const override { return THRESHOLD_STAGE; }
  std::string name() const override { return "Threshold"; }
  void forwardProp(std::shared_ptr<TorchData> input) override;

  float threshold;  // Single threshold value
  float val;        // Single output value (when input < threshold)

  static std::unique_ptr<TorchStage> loadFromFile(std::ifstream& file);

 protected:
  void init(std::shared_ptr<TorchData> input);

  // Non-copyable, non-assignable.
  Threshold(Threshold&);
  Threshold& operator=(const Threshold&);
};

};  // namespace jtorch
