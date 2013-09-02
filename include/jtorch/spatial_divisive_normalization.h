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

#include <mutex>
#include <condition_variable>
#include "jtil/math/math_types.h"
#include "jtil/threading/callback.h"
#include "jtorch/torch_stage.h"

namespace jtil { namespace data_str { template <typename T> class VectorManaged; } }

namespace jtorch {

  template <typename T> class Tensor;
  
  class SpatialDivisiveNormalization : public TorchStage {
  public:
    // Constructor / Destructor
    SpatialDivisiveNormalization(const Tensor<float>& kernel1d, 
      const float threshold = 1e-4f);
    virtual ~SpatialDivisiveNormalization();

    virtual TorchStageType type() const { return SPATIAL_DIVISIVE_NORMALIZATION_STAGE; }
    virtual void forwardProp(TorchData& input);

    static TorchStage* loadFromFile(std::ifstream& file);

  protected:
    Tensor<float>* kernel1d_;
    Tensor<float>* kernel1d_norm_;  // kernel normalization depends on input size
    Tensor<float>* std_coef_;
    Tensor<float>* std_;        // 2D
    Tensor<float>* std_pass1_;  // 3D - Horizontal pass
    Tensor<float>* std_pass2_;  // 3D - Vertical + normalization pass
    float threshold_;

    jtil::math::Int3 local_worgroup_size_3d;
    jtil::math::Int3 local_worgroup_size_2d;

    void init(TorchData& input);

    // Non-copyable, non-assignable.
    SpatialDivisiveNormalization(SpatialDivisiveNormalization&);
    SpatialDivisiveNormalization& operator=(const SpatialDivisiveNormalization&);
  };
  
};  // namespace jtorch
