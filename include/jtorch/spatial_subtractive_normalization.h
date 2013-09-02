//
//  spatial_subtractive_normalization.h
//
//  Created by Jonathan Tompson on 4/1/13.
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
  
  class SpatialSubtractiveNormalization : public TorchStage {
  public:
    // Constructor / Destructor
    SpatialSubtractiveNormalization(const Tensor<float>& kernel1d);
    virtual ~SpatialSubtractiveNormalization();

    virtual TorchStageType type() const { return SPATIAL_SUBTRACTIVE_NORMALIZATION_STAGE; }
    virtual void forwardProp(TorchData& input);

    static TorchStage* loadFromFile(std::ifstream& file);

  protected:
    Tensor<float>* kernel1d_;
    Tensor<float>* mean_coef_;
    Tensor<float>* mean_;        // 2D
    Tensor<float>* mean_pass1_;  // 3D - Horizontal pass
    Tensor<float>* mean_pass2_;  // 3D - Vertical + normalization pass

    jtil::math::Int3 local_worgroup_size_3d;
    jtil::math::Int3 local_worgroup_size_2d;

    void init(TorchData& input);

    // Non-copyable, non-assignable.
    SpatialSubtractiveNormalization(SpatialSubtractiveNormalization&);
    SpatialSubtractiveNormalization& operator=(const SpatialSubtractiveNormalization&);
  };
  
};  // namespace jtorch
