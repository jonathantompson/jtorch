//
//  tanh.h
//
//  Created by Jonathan Tompson on 4/1/13.
//

#pragma once

#include <mutex>
#include <condition_variable>
#include "jcl/math/int_types.h"
#include "jcl/math/math_types.h"
#include "jtorch/torch_stage.h"

namespace jcl { namespace data_str { template <typename T> class VectorManaged; } }

namespace jtorch {
  
  class Tanh : public TorchStage {
  public:
    // Constructor / Destructor
    Tanh();
    ~Tanh() override;

    TorchStageType type() const override { return TANH_STAGE; }
    std::string name() const override { return "Tanh"; }
    void forwardProp(std::shared_ptr<TorchData> input) override;

    static std::unique_ptr<TorchStage> loadFromFile(std::ifstream& file);

  protected:
    void init(std::shared_ptr<TorchData> input);

    // Non-copyable, non-assignable.
    Tanh(Tanh&);
    Tanh& operator=(const Tanh&);
  };
  
};  // namespace jtorch
