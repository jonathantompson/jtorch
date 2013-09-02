//
//  parallel.h
//
//  Created by Jonathan Tompson on 4/8/13.
//
//  Note: For now the output of each network in the parallel stage must be
//  a FloatTensor.
//

#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include "jtil/math/math_types.h"
#include "jtorch/torch_stage.h"

namespace jtil { namespace threading { class ThreadPool; } }
namespace jtil { namespace data_str { template <typename T> class VectorManaged; } }

namespace jtorch {
  
  class Parallel : public TorchStage {
  public:
    // Constructor / Destructor
    Parallel();
    virtual ~Parallel();

    virtual TorchStageType type() const { return PARALLEL_STAGE; }
    virtual void forwardProp(TorchData& input);

    void add(TorchStage* stage);

    uint32_t numBanks() const;

    TorchStage* get(const uint32_t i);

    static TorchStage* loadFromFile(std::ifstream& file);

  protected:
    jtil::data_str::VectorManaged<TorchStage*>* network_;

    void initOutput();

    // Non-copyable, non-assignable.
    Parallel(Parallel&);
    Parallel& operator=(const Parallel&);
  };

};  // namespace jtorch
