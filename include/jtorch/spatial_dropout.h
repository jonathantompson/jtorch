//
//  spatial_dropout.h
//
//  Created by Jonathan Tompson on 2/6/2015.
//
//  This is a feed forward (testing) only version.  No actual dropout is 
//  implemented.
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
  
  class SpatialDropout : public TorchStage {
  public:
    // Constructor / Destructor
    SpatialDropout(const float p);
    virtual ~SpatialDropout();

    virtual TorchStageType type() const { return SPATIAL_DROPOUT; }
    virtual std::string name() const { return "SpatialDropout"; }
    virtual void forwardProp(TorchData& input);

    static TorchStage* loadFromFile(std::ifstream& file);

  protected:
    float p_;

    void init(TorchData& input);

    // Non-copyable, non-assignable.
    SpatialDropout(SpatialDropout&);
    SpatialDropout& operator=(const SpatialDropout&);
  };
  
};  // namespace jtorch
