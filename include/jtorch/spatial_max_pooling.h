//
//  spatial_max_pooling.h
//
//  Created by Jonathan Tompson on 4/1/13.
//
//  Multithreading is not all that efficient:  Threads are split up per output
//  feature.
//

#pragma once

#include <mutex>
#include <condition_variable>
#include "jcl/math/int_types.h"
#include "jcl/math/math_types.h"
#include "jtorch/torch_stage.h"

namespace jcl {
namespace data_str {
template <typename T>
class VectorManaged;
}
}

namespace jtorch {

class SpatialMaxPooling : public TorchStage {
 public:
  // Constructor / Destructor
  SpatialMaxPooling(const uint32_t poolsize_v, const uint32_t poolsize_u);
  ~SpatialMaxPooling() override;

  TorchStageType type() const override { return SPATIAL_MAX_POOLING_STAGE; }
  std::string name() const override { return "SpatialMaxPooling"; }
  void forwardProp(std::shared_ptr<TorchData> input) override;

  static std::unique_ptr<TorchStage> loadFromFile(std::ifstream& file);

  inline uint32_t poolsize_u() const { return poolsize_u_; }
  inline uint32_t poolsize_v() const { return poolsize_v_; }

 protected:
  uint32_t poolsize_v_;
  uint32_t poolsize_u_;

  void init(std::shared_ptr<TorchData> input);

  // Non-copyable, non-assignable.
  SpatialMaxPooling(SpatialMaxPooling&);
  SpatialMaxPooling& operator=(const SpatialMaxPooling&);
};

};  // namespace jtorch
