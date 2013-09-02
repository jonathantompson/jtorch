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

#include <mutex>
#include <condition_variable>
#include "jtil/math/math_types.h"
#include "jtil/threading/callback.h"
#include "jtorch/torch_stage.h"

#define JTIL_SPATIAL_LP_POOLING_NTHREADS 4

namespace jtil { namespace data_str { template <typename T> class VectorManaged; } }

namespace jtorch {
  
  class SpatialLPPooling : public TorchStage {
  public:
    // Constructor / Destructor
    SpatialLPPooling(const float p_norm, const int32_t poolsize_v, 
      const int32_t poolsize_u);
    virtual ~SpatialLPPooling();

    virtual TorchStageType type() const { return SPATIAL_LP_POOLING_STAGE; }
    virtual void forwardProp(TorchData& input);

    static TorchStage* loadFromFile(std::ifstream& file);

  protected:
    float* input_cpu_;
    float* output_cpu_;
    int32_t cur_in_w;
    int32_t cur_in_h;
    float p_norm_;
    int32_t poolsize_v_;
    int32_t poolsize_u_;

    // Multithreading primatives and functions
    jtil::threading::ThreadPool* tp_;
    int32_t threads_finished_;
    std::mutex thread_update_lock_;
    std::condition_variable not_finished_;
    jtil::data_str::VectorManaged<jtil::threading::Callback<void>*>* thread_cbs_; 

    void forwardPropThread(const int32_t outf);

    void init(TorchData& input, jtil::threading::ThreadPool& tp);

    // Non-copyable, non-assignable.
    SpatialLPPooling(SpatialLPPooling&);
    SpatialLPPooling& operator=(const SpatialLPPooling&);
  };
  
};  // namespace jtorch
