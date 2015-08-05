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
  template <typename T> class Tensor;
  class Sequential;

  class SpatialContrastiveNormalization : public TorchStage {
  public:
    // Constructor / Destructor
    // Note if kernel is nullptr, then a rectangular filter kernel is used
    SpatialContrastiveNormalization(const Tensor<float>* kernel = nullptr, 
      const float threshold = 1e-4f);
    virtual ~SpatialContrastiveNormalization();

    virtual TorchStageType type() const { return SPATIAL_CONTRASTIVE_NORMALIZATION_STAGE; }
    virtual std::string name() const { return "SpatialContrastiveNormalization"; }
    virtual void forwardProp(TorchData& input);

    static TorchStage* loadFromFile(std::ifstream& file);

  protected:
    Sequential* network_;

    // Non-copyable, non-assignable.
    SpatialContrastiveNormalization(SpatialContrastiveNormalization&);
    SpatialContrastiveNormalization& operator=(const SpatialContrastiveNormalization&);
  };
  
};  // namespace jtorch
