//
//  tanh.h
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
  
  class Tanh : public TorchStage {
  public:
    // Constructor / Destructor
    Tanh();
    virtual ~Tanh();

    virtual TorchStageType type() const { return TANH_STAGE; }
    virtual std::string name() const { return "Tanh"; }
    virtual void forwardProp(TorchData& input);

    static TorchStage* loadFromFile(std::ifstream& file);

  protected:
    void init(TorchData& input);

    // Non-copyable, non-assignable.
    Tanh(Tanh&);
    Tanh& operator=(const Tanh&);
  };
  
};  // namespace jtorch
