//
//  select_table.h
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
  
  class SelectTable : public TorchStage {
  public:
    // Constructor / Destructor
    SelectTable(int32_t index);
    virtual ~SelectTable();

    virtual TorchStageType type() const { return SELECT_TABLE_STAGE; }
    virtual void forwardProp(TorchData& input);

    static TorchStage* loadFromFile(std::ifstream& file);

  protected:
    uint32_t index_;

    // Non-copyable, non-assignable.
    SelectTable(SelectTable&);
    SelectTable& operator=(const SelectTable&);
  };

};  // namespace jtorch
