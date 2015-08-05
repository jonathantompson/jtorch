//
//  join_table.h
//
//  Created by Jonathan Tompson on 4/9/13.
//
//  NOTE: This version of Join Table ALWAYS joins along the top dimension
//  As per the torch version, the dimension 0 is defined as the top most
//  dimension (ie f in fxhxw).
//

#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdint.h>
#include "jtorch/torch_stage.h"

namespace jtorch {
  
  class JoinTable : public TorchStage {
  public:
    // Constructor / Destructor
    JoinTable(const uint32_t dimension);
    ~JoinTable() override;

    TorchStageType type() const override { return JOIN_TABLE_STAGE; }
    std::string name() const override { return "JoinTable"; }
    void forwardProp(std::shared_ptr<TorchData> input) override;

    static std::unique_ptr<TorchStage> loadFromFile(std::ifstream& file);

    inline uint32_t dimension() const { return dimension_; }

  protected:
    void init(std::shared_ptr<TorchData> input);
    uint32_t dimension_;

    // Non-copyable, non-assignable.
    JoinTable(JoinTable&);
    JoinTable& operator=(const JoinTable&);
  };

};  // namespace jtorch
