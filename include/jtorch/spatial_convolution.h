//
//  spatial_convolution.h
//
//  Created by Jonathan Tompson on 5/15/13.
//

#pragma once

#include <mutex>
#include <condition_variable>
#include "jcl/math/int_types.h"
#include "jcl/math/math_types.h"
#include "jtorch/torch_stage.h"
#include "jcl/jcl.h"  // For jcl::JCLBuffer

namespace jcl { namespace data_str { template <typename T> class VectorManaged; } }

namespace jtorch {

  template <typename T> class Tensor;
  
  class SpatialConvolution : public TorchStage {
  public:
    // Constructor / Destructor
    SpatialConvolution(const int32_t feats_in, const int32_t feats_out,
      const int32_t filt_height, const int32_t filt_width);
    virtual ~SpatialConvolution();

    virtual TorchStageType type() const { return SPATIAL_CONVOLUTION_STAGE; }
    virtual void forwardProp(TorchData& input);

    void setWeights(const float* weights);
    void setBiases(const float* biases);
    Tensor<float>* weights() { return weights_; }
    Tensor<float>* biases() { return biases_; }

    static TorchStage* loadFromFile(std::ifstream& file);

  protected:
    int32_t filt_width_;
    int32_t filt_height_;
    int32_t feats_in_;
    int32_t feats_out_;

    // weights_buf_:    dim[2] --> matrix_index (size = feats_out_ * feats_in) 
    //                  dim[1] --> filter height
    //                  dim[0] --> filter width
    Tensor<float>* weights_;
    // biases_buf_:     dim[0] --> feats_out_t
    Tensor<float>* biases_;

    jcl::math::Int3 local_worgroup_size;

    void init(TorchData& input);

    // Non-copyable, non-assignable.
    SpatialConvolution(SpatialConvolution&);
    SpatialConvolution& operator=(const SpatialConvolution&);
  };
  
};  // namespace jtorch
