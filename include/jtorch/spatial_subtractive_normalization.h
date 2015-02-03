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
    SpatialSubtractiveNormalization(const Tensor<float>& kernel);
    virtual ~SpatialSubtractiveNormalization();

    virtual TorchStageType type() const { return SPATIAL_SUBTRACTIVE_NORMALIZATION_STAGE; }
    virtual void forwardProp(TorchData& input);

    static TorchStage* loadFromFile(std::ifstream& file);

  protected:
    Tensor<float>* kernel_;
    Tensor<float>* mean_coef_;
    Tensor<float>* mean_;        // 2D
    Tensor<float>* mean_pass1_;  // 3D - Horizontal pass
    Tensor<float>* mean_pass2_;  // 3D - Vertical + normalization pass

    void init(TorchData& input);
    void cleanup();

    // Non-copyable, non-assignable.
    SpatialSubtractiveNormalization(SpatialSubtractiveNormalization&);
    SpatialSubtractiveNormalization& operator=(const SpatialSubtractiveNormalization&);
  };
  
};  // namespace jtorch
