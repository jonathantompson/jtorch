//
//  threshold.h
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
  
  class Threshold : public TorchStage {
  public:
    // Constructor / Destructor
    Threshold();
    virtual ~Threshold();

    virtual TorchStageType type() const { return THRESHOLD_STAGE; }
    virtual void forwardProp(TorchData& input);

    float threshold;  // Single threshold value
    float val;  // Single output value (when input < threshold)

    static TorchStage* loadFromFile(std::ifstream& file);

  protected:
    void init(TorchData& input);
    jcl::math::Int3 local_worgroup_size;

    // Non-copyable, non-assignable.
    Threshold(Threshold&);
    Threshold& operator=(const Threshold&);
  };
  
};  // namespace jtorch
