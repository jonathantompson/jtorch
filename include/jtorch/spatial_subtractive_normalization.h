//
//  spatial_subtractive_normalization.h
//
//  Created by Jonathan Tompson on 4/1/13.
//

#pragma once

#include <mutex>
#include <condition_variable>
#include "jcl/math/int_types.h"
#include "jcl/math/math_types.h"
#include "jtorch/torch_stage.h"

namespace jcl { namespace data_str { template <typename T> class VectorManaged; } }

namespace jtorch {

  template <typename T> class Tensor;
  
  class SpatialSubtractiveNormalization : public TorchStage {
  public:
    // Constructor / Destructor
    SpatialSubtractiveNormalization(const std::shared_ptr<Tensor<float>> kernel);
    ~SpatialSubtractiveNormalization() override;

    TorchStageType type() const override { return SPATIAL_SUBTRACTIVE_NORMALIZATION_STAGE; }
    std::string name() const override { return "SpatialSubtractiveNormalization"; }
    void forwardProp(std::shared_ptr<TorchData> input) override;

    static std::unique_ptr<TorchStage> loadFromFile(std::ifstream& file);

  protected:
    std::unique_ptr<Tensor<float>> kernel_;
    std::unique_ptr<Tensor<float>> mean_coef_;
    std::unique_ptr<Tensor<float>> mean_;        // 2D
    std::unique_ptr<Tensor<float>> mean_pass1_;  // 3D - Horizontal pass
    std::unique_ptr<Tensor<float>> mean_pass2_;  // 3D - Vertical + normalization pass

    void init(std::shared_ptr<TorchData> input);
    void cleanup();

    // Non-copyable, non-assignable.
    SpatialSubtractiveNormalization(SpatialSubtractiveNormalization&);
    SpatialSubtractiveNormalization& operator=(const SpatialSubtractiveNormalization&);
  };
  
};  // namespace jtorch
