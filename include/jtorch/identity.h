//
//  identity.h
//
//  Created by Jonathan Tompson on 1/26/15.
//
//  Identity just passes the data through
//

#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include "jtorch/torch_stage.h"

namespace jtorch {
  
  class Identity : public TorchStage {
  public:
    // Constructor / Destructor
    Identity();
    virtual ~Identity();

    virtual TorchStageType type() const { return IDENTITY_STAGE; }
    virtual void forwardProp(TorchData& input);

    static TorchStage* loadFromFile(std::ifstream& file);

  protected:

    // Non-copyable, non-assignable.
    Identity(Identity&);
    Identity& operator=(const Identity&);
  };

};  // namespace jtorch
