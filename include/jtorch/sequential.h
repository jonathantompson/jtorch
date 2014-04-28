//
//  sequential.h
//
//  Created by Jonathan Tompson on 4/2/13.
//

#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include "jcl/math/int_types.h"
#include "jtorch/torch_stage.h"

namespace jcl { namespace data_str { template <typename T> class VectorManaged; } }

namespace jtorch {
  
  class Sequential : public TorchStage {
  public:
    // Constructor / Destructor
    Sequential();
    virtual ~Sequential();

    virtual TorchStageType type() const { return SEQUENTIAL_STAGE; }
    virtual void forwardProp(TorchData& input);

    void add(TorchStage* stage);
    TorchStage* get(const uint32_t i);
    uint32_t size() const;

    static TorchStage* loadFromFile(std::ifstream& file);

  protected:
    jcl::data_str::VectorManaged<TorchStage*>* network_;

    // Non-copyable, non-assignable.
    Sequential(Sequential&);
    Sequential& operator=(const Sequential&);
  };

};  // namespace jtorch
