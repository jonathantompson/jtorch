//
//  spatial_contrastive_normalization.h
//
//  Created by Jonathan Tompson on 4/1/13.
//
//  This stage is only partially multi-threaded!
//
//  It is just a SpatialSubtractiveNormalization followed by a
//  SpatialDivisiveNormalization.  In other words, subtracting off the mean and
//  dividing by the standard deviation.
//
//  This stage is the default for local contrast normalization.
//

#pragma once

#include <mutex>
#include <condition_variable>
#include "jtorch/torch_stage.h"

namespace jtorch {
template <typename T>
class Tensor;
class Sequential;

class SpatialContrastiveNormalization : public TorchStage {
 public:
  // Constructor / Destructor
  // Note if kernel is nullptr, then a rectangular filter kernel is used
  SpatialContrastiveNormalization(
      std::shared_ptr<Tensor<float>> kernel = nullptr, float threshold = 1e-4f);
  ~SpatialContrastiveNormalization() override;

  TorchStageType type() const override {
    return SPATIAL_CONTRASTIVE_NORMALIZATION_STAGE;
  }
  std::string name() const override {
    return "SpatialContrastiveNormalization";
  }
  void forwardProp(std::shared_ptr<TorchData> input) override;

  static std::unique_ptr<TorchStage> loadFromFile(std::ifstream& file);

 protected:
  std::unique_ptr<Sequential> network_;

  // Non-copyable, non-assignable.
  SpatialContrastiveNormalization(SpatialContrastiveNormalization&);
  SpatialContrastiveNormalization& operator=(
      const SpatialContrastiveNormalization&);
};

};  // namespace jtorch
