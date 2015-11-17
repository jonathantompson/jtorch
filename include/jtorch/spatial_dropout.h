//
//  spatial_dropout.h
//
//  Created by Jonathan Tompson on 2/6/2015.
//
//  This is a feed forward (testing) only version.  No actual dropout is
//  implemented.
//

#pragma once

#include "jcl/math/int_types.h"
#include "jcl/math/math_types.h"
#include "jtorch/torch_stage.h"

namespace jtorch {

template <typename T>
class Tensor;

class SpatialDropout : public TorchStage {
 public:
  // Constructor / Destructor
  SpatialDropout(const float p);
  ~SpatialDropout() override;

  TorchStageType type() const override { return SPATIAL_DROPOUT_STAGE; }
  std::string name() const override { return "SpatialDropout"; }
  void forwardProp(std::shared_ptr<TorchData> input) override;

  static std::unique_ptr<TorchStage> loadFromFile(std::ifstream& file);

 protected:
  float p_;

  void init(std::shared_ptr<TorchData> input);

  // Non-copyable, non-assignable.
  SpatialDropout(const SpatialDropout&) = delete;
  SpatialDropout& operator=(const SpatialDropout&) = delete;
};

};  // namespace jtorch
