//
//  spatial_up_sampling_nearest.h
//
//  Created by Jonathan Tompson on 1/27/15.
//

#pragma once

#include <mutex>
#include <condition_variable>
#include "jcl/math/int_types.h"
#include "jcl/math/math_types.h"
#include "jtorch/torch_stage.h"
#include "jcl/jcl.h"  // For jcl::JCLBuffer

namespace jcl { namespace data_str { template <typename T> class VectorManaged; } }

namespace jtorch {

  template <typename T> class Tensor;
  
  class SpatialUpSamplingNearest : public TorchStage {
  public:
    // Constructor / Destructor
    SpatialUpSamplingNearest(const int32_t scale);
    virtual ~SpatialUpSamplingNearest();

    virtual TorchStageType type() const { return SPATIAL_UP_SAMPLING_NEAREST_STAGE; }
    virtual std::string name() const { return "SpatialUpSamplingNearest"; }
    virtual void forwardProp(TorchData& input);

    static TorchStage* loadFromFile(std::ifstream& file);

  protected:
    uint32_t scale_;
    uint32_t out_dim_;
    uint32_t* out_size_;

    void init(TorchData& input);

    // Non-copyable, non-assignable.
    SpatialUpSamplingNearest(SpatialUpSamplingNearest&);
    SpatialUpSamplingNearest& operator=(const SpatialUpSamplingNearest&);
  };
  
};  // namespace jtorch
