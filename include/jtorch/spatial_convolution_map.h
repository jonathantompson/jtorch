//
//  spatial_convolution_map.h
//
//  Created by Jonathan Tompson on 4/1/13.
//
//  Same as SpatialConvolution except that the output features aren't fully
//  connected to the input features (so we need to keep around a connection
//  table).
//
//  Multithreading is NOT all that efficient:  Threads are split up per output 
//  feature.  This has not been implemented in OpenCL yet (since I no longer
//  use this stage).
//

#pragma once

#include <mutex>
#include <condition_variable>
#include "jcl/math/int_types.h"
#include "jcl/threading/callback.h"
#include "jtorch/torch_stage.h"

#define JTIL_SPATIAL_CONVOLUTION_MAP_NTHREADS 4

namespace jcl { namespace data_str { template <typename T> class VectorManaged; } }
namespace jcl { namespace threading { class ThreadPool; } }

namespace jtorch {
  
  class SpatialConvolutionMap : public TorchStage {
  public:
    // Constructor / Destructor
    SpatialConvolutionMap(const uint32_t feats_in, const uint32_t feats_out,
      const uint32_t fan_in, const uint32_t filt_height, 
      const uint32_t filt_width);
    virtual ~SpatialConvolutionMap();

    virtual TorchStageType type() const { return SPATIAL_CONVOLUTION_MAP_STAGE; }
    virtual std::string name() const { return "SpatialConvolutionMap"; }
    virtual void forwardProp(TorchData& input);

    float** weights;
    float* biases;
    int16_t** conn_table;  // This is the same as conn_table_rev in Torch
                           // For each output: [0] is input feature and [1]
                           // is the weight matrix (filter) to use

    static TorchStage* loadFromFile(std::ifstream& file);

  protected:
    float* input_cpu_;
    float* output_cpu_;
    uint32_t filt_width_;
    uint32_t filt_height_;
    uint32_t feats_in_;
    uint32_t feats_out_;
    uint32_t fan_in_;

    // Multithreading primatives and functions
    jcl::threading::ThreadPool* tp_;
    float* cur_input_;
    int32_t cur_input_width_;
    int32_t cur_input_height_;
    float* cur_output_;
    int32_t threads_finished_;
    std::mutex thread_update_lock_;
    std::condition_variable not_finished_;
    jcl::data_str::VectorManaged<jcl::threading::Callback<void>*>* thread_cbs_; 

    void forwardPropThread(const uint32_t outf);

    void init(TorchData& input, jcl::threading::ThreadPool& tp);

    // Non-copyable, non-assignable.
    SpatialConvolutionMap(SpatialConvolutionMap&);
    SpatialConvolutionMap& operator=(const SpatialConvolutionMap&);
  };
  
};  // namespace jtorch
