//
//  spatial_divisive_normalization.h
//
//  Created by Jonathan Tompson on 4/1/13.
//
//  Note that just like the torch stage, this divisive stage assumes zero
//  input mean.  That is, it does not subtract off the mean per element when
//  estimating the standard deviation.
//

#pragma once

#include "jcl/math/int_types.h"
#include "jcl/math/math_types.h"
#include "jtorch/torch_stage.h"

namespace jtorch {

template <typename T>
class Tensor;

class SpatialDivisiveNormalization : public TorchStage {
 public:
  // Constructor / Destructor
  SpatialDivisiveNormalization(const std::shared_ptr<Tensor<float>> kernel,
                               const float threshold = 1e-4f);
  ~SpatialDivisiveNormalization() override;

  TorchStageType type() const override {
    return SPATIAL_DIVISIVE_NORMALIZATION_STAGE;
  }
  std::string name() const override { return "SpatialDivisiveNormalization"; }
  void forwardProp(std::shared_ptr<TorchData> input) override;

  static std::unique_ptr<TorchStage> loadFromFile(std::ifstream& file);

 protected:
  std::shared_ptr<Tensor<float>> kernel_;
  std::shared_ptr<Tensor<float>>
      kernel_norm_;  // kernel normalization depends on input size
  std::shared_ptr<Tensor<float>> std_coef_;
  std::shared_ptr<Tensor<float>> std_;        // 2D
  std::shared_ptr<Tensor<float>> std_pass1_;  // 3D - Horizontal pass
  std::shared_ptr<Tensor<float>>
      std_pass2_;  // 3D - Vertical + normalization pass
  float threshold_;

  void init(std::shared_ptr<TorchData> input);
  void cleanup();

  // Non-copyable, non-assignable.
  SpatialDivisiveNormalization(const SpatialDivisiveNormalization&) = delete;
  SpatialDivisiveNormalization& operator=(const SpatialDivisiveNormalization&) =
      delete;
};

};  // namespace jtorch
