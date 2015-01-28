//
//  parallel_table.h
//
//  Created by Jonathan Tompson on 4/8/13.
//

#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include "jcl/math/int_types.h"
#include "jtorch/torch_stage.h"

namespace jcl { namespace data_str { template <typename T> class VectorManaged; } }

namespace jtorch {
  
  class ParallelTable : public TorchStage {
  public:
    // Constructor / Destructor
    ParallelTable();
    virtual ~ParallelTable();

    virtual TorchStageType type() const { return PARALLEL_TABLE_STAGE; }
    virtual void forwardProp(TorchData& input);

    void add(TorchStage* stage);

    uint32_t numBanks() const;

    TorchStage* get(const uint32_t i);

    static TorchStage* loadFromFile(std::ifstream& file);

  protected:
    jcl::data_str::VectorManaged<TorchStage*>* network_;

    void initOutput();

    // Non-copyable, non-assignable.
    ParallelTable(ParallelTable&);
    ParallelTable& operator=(const ParallelTable&);
  };

};  // namespace jtorch
