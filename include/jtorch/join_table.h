//
//  join_table.h
//
//  Created by Jonathan Tompson on 4/9/13.
//
//  NOTE: This version of Join Table ALWAYS joins along dimension 0
//

#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include "jtorch/torch_stage.h"

namespace jtorch {
  
  class JoinTable : public TorchStage {
  public:
    // Constructor / Destructor
    JoinTable();
    virtual ~JoinTable();

    virtual TorchStageType type() const { return JOIN_TABLE_STAGE; }
    virtual void forwardProp(TorchData& input);

    static TorchStage* loadFromFile(std::ifstream& file);

  protected:
    void init(TorchData& input);

    // Non-copyable, non-assignable.
    JoinTable(JoinTable&);
    JoinTable& operator=(const JoinTable&);
  };

};  // namespace jtorch
