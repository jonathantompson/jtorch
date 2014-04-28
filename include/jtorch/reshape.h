//
//  reshape.h
//
//  Created by Jonathan Tompson on 4/1/13.
//
//  Works a little differently to the torch version...  It takes a 3D tensor
//  and just makes it into a 1D array, so it's not as general purpose.
//
//  But really this 1D array is just a straight copy of the input data (since
//  we define tensors as float* anyway).
//

#pragma once

#include <mutex>
#include <condition_variable>
#include "jtorch/torch_stage.h"

namespace jtorch {
  
  class Reshape : public TorchStage {
  public:
    // Constructor / Destructor
    Reshape();
    virtual ~Reshape();

    virtual TorchStageType type() const { return RESHAPE_STAGE; }
    virtual void forwardProp(TorchData& input);

    static TorchStage* loadFromFile(std::ifstream& file);

  protected:
    void init(TorchData& input);

    // Non-copyable, non-assignable.
    Reshape(Reshape&);
    Reshape& operator=(const Reshape&);
  };
  
};  // namespace jtorch
