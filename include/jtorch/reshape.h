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
#include "jcl/math/math_types.h"

namespace jtorch {
  
  class Reshape : public TorchStage {
  public:
    // Constructor / Destructor
    // For 1D tensor: set sz1 = -1, for 2D tensor: set sz2 = -1
    Reshape(const uint32_t dim, const uint32_t* size);
    virtual ~Reshape();

    virtual TorchStageType type() const { return RESHAPE_STAGE; }
    virtual void forwardProp(TorchData& input);

    static TorchStage* loadFromFile(std::ifstream& file);

  protected:
    uint32_t odim_;
    uint32_t* osize_;
    void init(TorchData& input);

    uint32_t outNElem() const;

    // Non-copyable, non-assignable.
    Reshape(Reshape&);
    Reshape& operator=(const Reshape&);
  };
  
};  // namespace jtorch
