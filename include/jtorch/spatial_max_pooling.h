//
//  spatial_max_pooling.h
//
//  Created by Jonathan Tompson on 4/1/13.
//
//  Multithreading is not all that efficient:  Threads are split up per output 
//  feature.
//

#pragma once

#include <mutex>
#include <condition_variable>
#include "jtil/math/math_types.h"
#include "jtil/threading/callback.h"
#include "jtorch/torch_stage.h"

namespace jtil { namespace data_str { template <typename T> class VectorManaged; } }

namespace jtorch {
  
  class SpatialMaxPooling : public TorchStage {
  public:
    // Constructor / Destructor
    SpatialMaxPooling(const int32_t poolsize_v, const int32_t poolsize_u);
    virtual ~SpatialMaxPooling();

    virtual TorchStageType type() const { return SPATIAL_MAX_POOLING_STAGE; }
    virtual void forwardProp(TorchData& input);

    static TorchStage* loadFromFile(std::ifstream& file);

  protected:
    int32_t poolsize_v_;
    int32_t poolsize_u_;

    jtil::math::Int3 local_worgroup_size;
    void init(TorchData& input);

    // Non-copyable, non-assignable.
    SpatialMaxPooling(SpatialMaxPooling&);
    SpatialMaxPooling& operator=(const SpatialMaxPooling&);
  };
  
};  // namespace jtorch
