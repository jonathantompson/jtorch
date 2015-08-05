#include "jtorch/spatial_lp_pooling.h"
#include "jtorch/tensor.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"
#include "jcl/data_str/vector_managed.h"

#define SAFE_DELETE(x) if (x != nullptr) { delete x; x = nullptr; }
#define SAFE_DELETE_ARR(x) if (x != nullptr) { delete[] x; x = nullptr; }

using namespace jcl::threading;
using namespace jcl::math;
using namespace jcl::data_str;

namespace jtorch {

  SpatialLPPooling::SpatialLPPooling(const float p_norm, 
    const uint32_t poolsize_v, const uint32_t poolsize_u) : TorchStage() {
    p_norm_ = p_norm;
    poolsize_v_ = poolsize_v;
    poolsize_u_ = poolsize_u;
    output = nullptr;
    thread_cbs_ = nullptr;
    output_cpu_ = nullptr;
    input_cpu_ = nullptr;

    tp_ = new ThreadPool(JTIL_SPATIAL_LP_POOLING_NTHREADS);

    std::cout << "WARNING: SPATIALLPPOOLING IS SLOW." << std::endl;
    std::cout << "--> ALL COMPUTATION IS DONE ON THE CPU!" << std::endl;
  }

  void SpatialLPPooling::cleanup() {
    SAFE_DELETE_ARR(output_cpu_);
    SAFE_DELETE_ARR(input_cpu_);
    SAFE_DELETE(output);
    SAFE_DELETE(thread_cbs_);
  }

  SpatialLPPooling::~SpatialLPPooling() {
    tp_->stop();
    SAFE_DELETE(tp_);
    cleanup();
  }

  void SpatialLPPooling::init(TorchData& input, ThreadPool& tp)  {
    if (input.type() != TorchDataType::TENSOR_DATA) {
      throw std::runtime_error("SpatialLPPooling::init() - "
        "FloatTensor expected!");
    }
    Tensor<float>& in = (Tensor<float>&)input;
    if (in.dim() != 2 && in.dim() != 3) {
      throw std::runtime_error("Input dimension must be 2D or 3D!");
    }

    if (output != nullptr && TO_TENSOR_PTR(output)->dim() != in.dim()) {
      // Input dimension has changed!
      cleanup();
    }

    if (output != nullptr) {
      // Check that the dimensions above the lowest 2 match
      for (uint32_t i = 2; i < in.dim() && output != nullptr; i++) {
        if (TO_TENSOR_PTR(output)->size()[i] != in.size()[i]) {
          cleanup();
        }
      }
    }

    if (output != nullptr) {
      // Check that the lowest 2 dimensions are the correct size
      if (TO_TENSOR_PTR(output)->size()[0] != in.size()[0] / poolsize_u_ ||
        TO_TENSOR_PTR(output)->size()[1] != in.size()[1] / poolsize_v_) {
        cleanup();
      }
    }

    if (output == nullptr) {
      if (in.size()[0] % poolsize_u_ != 0 || 
          in.size()[1] % poolsize_v_ != 0) {
        throw std::runtime_error("width or height is not a multiple of "
          "the poolsize!");
      }

      uint32_t* out_size = new uint32_t[in.dim()];
      out_size[0] = in.size()[0] / poolsize_u_;
      out_size[1] = in.size()[1] / poolsize_v_;
      for (uint32_t i = 2; i < in.dim(); i++) {
        out_size[i] = in.size()[i];
      }

      output = new Tensor<float>(in.dim(), out_size);
      input_cpu_ = new float[in.nelems()];
      output_cpu_ = new float[TO_TENSOR_PTR(output)->nelems()];
      SAFE_DELETE_ARR(out_size);
    }

    if (thread_cbs_ == nullptr) {
      uint32_t n_threads = 1;
      if (in.dim() > 2) {
        n_threads = TO_TENSOR_PTR(output)->size()[2];
      }
      thread_cbs_ = new VectorManaged<Callback<void>*>(n_threads);
      for (uint32_t f = 0; f < n_threads; f++) {
        thread_cbs_->pushBack(MakeCallableMany(
          &SpatialLPPooling::forwardPropThread, this, f));
      }
    }
  }

  void SpatialLPPooling::forwardProp(TorchData& input) { 
    init(input, *tp_);
    Tensor<float>*in = &((Tensor<float>&)input);
    in->getData(input_cpu_);
    cur_in_w = in->size()[0];
    cur_in_h = in->size()[1];
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
    TO_TENSOR_PTR(output)->setData(output_cpu_);
  }

  void SpatialLPPooling::forwardPropThread(const uint32_t outf) {
    const uint32_t out_w = TO_TENSOR_PTR(output)->size()[0];
    const uint32_t out_h = TO_TENSOR_PTR(output)->size()[1];
    const uint32_t in_w = cur_in_w;
    const uint32_t in_h = cur_in_h;
    const float one_over_p_norm = 1.0f / p_norm_;

    float* out = &output_cpu_[outf * out_w * out_h];
    float* in = &input_cpu_[outf * in_w * in_h];
      
    for (uint32_t outv = 0; outv < out_h; outv++) {
      for (uint32_t outu = 0; outu < out_w; outu++) {
        uint32_t out_index = outv * out_w + outu;
        out[out_index] = 0.0f;
        // Now perform max pooling:
        for (uint32_t inv = outv * poolsize_v_; inv < (outv + 1) * poolsize_v_; inv++) {
          for (uint32_t inu = outu * poolsize_u_; inu < (outu + 1) * poolsize_u_; inu++) {
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
