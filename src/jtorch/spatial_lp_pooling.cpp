#include "jtorch/spatial_lp_pooling.h"
#include "jtorch/tensor.h"
#include "jtil/exceptions/wruntime_error.h"
#include "jtil/threading/thread.h"
#include "jtil/threading/callback.h"
#include "jtil/threading/thread_pool.h"
#include "jtil/data_str/vector_managed.h"

#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }

using namespace jtil::threading;
using namespace jtil::math;
using namespace jtil::data_str;

namespace jtorch {

  SpatialLPPooling::SpatialLPPooling(const float p_norm, 
    const int32_t poolsize_v, const int32_t poolsize_u) : TorchStage() {
    p_norm_ = p_norm;
    poolsize_v_ = poolsize_v;
    poolsize_u_ = poolsize_u;
    output = NULL;
    thread_cbs_ = NULL;
    output_cpu_ = NULL;
    input_cpu_ = NULL;

    tp_ = new ThreadPool(JTIL_SPATIAL_LP_POOLING_NTHREADS);

    std::cout << "WARNING: SPATIALLPPOOLING IS SLOW.  All computation is";
    std::cout << " done on the CPU and incurs large transfer penalties!";
    std::cout << std::endl;
  }

  SpatialLPPooling::~SpatialLPPooling() {
    tp_->stop();
    SAFE_DELETE(tp_);
    SAFE_DELETE_ARR(output_cpu_);
    SAFE_DELETE_ARR(input_cpu_);
    SAFE_DELETE(output);
    SAFE_DELETE(thread_cbs_);
  }

  void SpatialLPPooling::init(TorchData& input, ThreadPool& tp)  {
    if (input.type() != TorchDataType::TENSOR_DATA) {
      throw std::wruntime_error("SpatialLPPooling::init() - "
        "FloatTensor expected!");
    }
    Tensor<float>& in = (Tensor<float>&)input;
    if (output != NULL) {
      if (!Int3::equal(in.dim(), ((Tensor<float>*)output)->dim())) {
        // Input dimension has changed!
        SAFE_DELETE(output);
        SAFE_DELETE(thread_cbs_);
        SAFE_DELETE_ARR(output_cpu_);
        SAFE_DELETE_ARR(input_cpu_);
      }
    }
    if (output == NULL) {
      if (in.dim()[0] % poolsize_u_ != 0 || 
        in.dim()[1] % poolsize_v_ != 0) {
        throw std::wruntime_error("width or height is not a multiple of "
          "the poolsize!");
      }
      Int3 out_dim(in.dim());
      out_dim[0] /= poolsize_u_;
      out_dim[1] /= poolsize_v_;
      output = new Tensor<float>(out_dim);
      input_cpu_ = new float[input.dataSize()];
      output_cpu_ = new float[output->dataSize()];
    }

    if (thread_cbs_ == NULL) {
      int32_t n_threads = ((Tensor<float>*)output)->dim()[2];
      thread_cbs_ = new VectorManaged<Callback<void>*>(n_threads);
      for (int32_t f = 0; f < ((Tensor<float>*)output)->dim()[2]; f++) {
        thread_cbs_->pushBack(MakeCallableMany(
          &SpatialLPPooling::forwardPropThread, this, f));
      }
    }
  }

  void SpatialLPPooling::forwardProp(TorchData& input) { 
    init(input, *tp_);
    Tensor<float>*in = &((Tensor<float>&)input);
    in->getData(input_cpu_);
    cur_in_w = in->dim()[0];
    cur_in_h = in->dim()[1];
    threads_finished_ = 0;
    for (uint32_t i = 0; i < thread_cbs_->size(); i++) {
      tp_->addTask((*thread_cbs_)[i]);
    } 

    // Wait for all threads to finish
    std::unique_lock<std::mutex> ul(thread_update_lock_);  // Get lock
    while (threads_finished_ != static_cast<int32_t>(thread_cbs_->size())) {
      not_finished_.wait(ul);
    }
    ul.unlock();  // Release lock
    ((Tensor<float>*)output)->setData(output_cpu_);
  }

  void SpatialLPPooling::forwardPropThread(const int32_t outf) {
    const int32_t out_w = ((Tensor<float>*)output)->dim()[0];
    const int32_t out_h = ((Tensor<float>*)output)->dim()[1];
    const int32_t in_w = cur_in_w;
    const int32_t in_h = cur_in_h;
    const float one_over_p_norm = 1.0f / p_norm_;

    float* out = &output_cpu_[outf * out_w * out_h];
    float* in = &input_cpu_[outf * in_w * in_h];
      
    for (int32_t outv = 0; outv < out_h; outv++) {
      for (int32_t outu = 0; outu < out_w; outu++) {
        int32_t out_index = outv * out_w + outu;
        out[out_index] = 0.0f;
        // Now perform max pooling:
        for (int32_t inv = outv * poolsize_v_; inv < (outv + 1) * poolsize_v_; inv++) {
          for (int32_t inu = outu * poolsize_u_; inu < (outu + 1) * poolsize_u_; inu++) {
            float val = fabsf(in[inv * in_w + inu]);
            out[out_index] += powf(val, p_norm_);
          }
        }
        out[outv * out_w + outu] = powf(out[outv * out_w + outu], 
          one_over_p_norm);
      }
    }
    std::unique_lock<std::mutex> ul(thread_update_lock_);
    threads_finished_++;
    not_finished_.notify_all();  // Signify that all threads might have finished
    ul.unlock();
  }

  TorchStage* SpatialLPPooling::loadFromFile(std::ifstream& file) {
    int filt_width, filt_height, pnorm;
    file.read((char*)(&filt_width), sizeof(filt_width));
    file.read((char*)(&filt_height), sizeof(filt_height));
    file.read((char*)(&pnorm), sizeof(pnorm));
    return new SpatialLPPooling((float)pnorm, filt_height, filt_width);
  }

}  // namespace jtorch
