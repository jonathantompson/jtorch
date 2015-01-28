//
//  c_add_table.h
//
//  Created by Jonathan Tompson on 1/27/15.
//

#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdint.h>
#include "jtorch/torch_stage.h"

namespace jtorch {
  
  class CAddTable : public TorchStage {
  public:
    // Constructor / Destructor
    CAddTable();
    virtual ~CAddTable();

    virtual TorchStageType type() const { return C_ADD_TABLE_STAGE; }
    virtual void forwardProp(TorchData& input);

    static TorchStage* loadFromFile(std::ifstream& file);

  protected:

    // Non-copyable, non-assignable.
    CAddTable(CAddTable&);
    CAddTable& operator=(const CAddTable&);
  };

};  // namespace jtorch
