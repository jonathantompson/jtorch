//
//  spatial_lp_pooling.h
//
//  Created by Jonathan Tompson on 4/1/13.
//
//  Multithreading is NOT all that efficient:  Threads are split up per output
//  feature.  This has not been implemented in OpenCL yet (since I no longer
//  use this stage).
//

#pragma once

#include <condition_variable>
#include <mutex>
#include <vector>
#include "jcl/math/int_types.h"
#include "jcl/threading/callback.h"
#include "jtorch/torch_stage.h"

#define JTIL_SPATIAL_LP_POOLING_NTHREADS 4

namespace jcl {
namespace data_str {
template <typename T>
class VectorManaged;
}
}
namespace jcl {
namespace threading {
class ThreadPool;
}
}

namespace jtorch {

class SpatialLPPooling : public TorchStage {
 public:
  // Constructor / Destructor
  SpatialLPPooling(const float p_norm, const uint32_t poolsize_v,
                   const uint32_t poolsize_u);
  ~SpatialLPPooling() override;

  TorchStageType type() const override { return SPATIAL_LP_POOLING_STAGE; }
  std::string name() const override { return "SpatialLPPooling"; }
  void forwardProp(std::shared_ptr<TorchData> input) override;

  static std::unique_ptr<TorchStage> loadFromFile(std::ifstream& file);

 protected:
  std::unique_ptr<float[]> input_cpu_;
  std::unique_ptr<float[]> output_cpu_;
  uint32_t cur_in_w;
  uint32_t cur_in_h;
  float p_norm_;
  uint32_t poolsize_v_;
  uint32_t poolsize_u_;

  // Multithreading primatives and functions
  std::unique_ptr<jcl::threading::ThreadPool> tp_;
  int32_t threads_finished_;
  std::mutex thread_update_lock_;
  std::condition_variable not_finished_;
  std::vector<std::unique_ptr<jcl::threading::Callback<void>>> thread_cbs_;

  void forwardPropThread(const uint32_t outf);

  void init(std::shared_ptr<TorchData> input);
  void cleanup();

  // Non-copyable, non-assignable.
  SpatialLPPooling(SpatialLPPooling&);
  SpatialLPPooling& operator=(const SpatialLPPooling&);
};

};  // namespace jtorch
