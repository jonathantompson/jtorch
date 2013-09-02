//
//  tanh.h
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
  
  class Tanh : public TorchStage {
  public:
    // Constructor / Destructor
    Tanh();
    virtual ~Tanh();

    virtual TorchStageType type() const { return TANH_STAGE; }
    virtual void forwardProp(TorchData& input);

    static TorchStage* loadFromFile(std::ifstream& file);

  protected:
    void init(TorchData& input);
    jtil::math::Int3 local_worgroup_size;

    // Non-copyable, non-assignable.
    Tanh(Tanh&);
    Tanh& operator=(const Tanh&);
  };
  
};  // namespace jtorch
