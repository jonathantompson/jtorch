#include "jtorch/spatial_lp_pooling.h"
#include "jtorch/tensor.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"
#include "jcl/data_str/vector_managed.h"

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
    output_cpu_.reset(nullptr);
    input_cpu_.reset(nullptr);

    tp_.reset(new ThreadPool(JTIL_SPATIAL_LP_POOLING_NTHREADS));

    std::cout << "WARNING: SPATIALLPPOOLING IS SLOW." << std::endl;
    std::cout << "--> ALL COMPUTATION IS DONE ON THE CPU!" << std::endl;
  }

  void SpatialLPPooling::cleanup() {
    output_cpu_.reset(nullptr);
    input_cpu_.reset(nullptr);
    output = nullptr;
    thread_cbs_.clear();
  }

  SpatialLPPooling::~SpatialLPPooling() {
    tp_->stop();
    cleanup();
  }

  void SpatialLPPooling::init(std::shared_ptr<TorchData> input)  {
    assert(input->type() == TorchDataType::TENSOR_DATA);
    Tensor<float>* in = TO_TENSOR_PTR(input.get());
    assert(in->dim() == 2 || in->dim() == 3);

    if (output != nullptr && 
        TO_TENSOR_PTR(output.get())->dim() != in->dim()) {
      // Input dimension has changed!
      cleanup();
    }

    if (output != nullptr) {
      // Check that the dimensions above the lowest 2 match
      for (uint32_t i = 2; i < in->dim() && output != nullptr; i++) {
        if (TO_TENSOR_PTR(output.get())->size()[i] != in->size()[i]) {
          cleanup();
        }
      }
    }

    if (output != nullptr) {
      // Check that the lowest 2 dimensions are the correct size
      if (TO_TENSOR_PTR(output.get())->size()[0] != in->size()[0] / poolsize_u_ ||
        TO_TENSOR_PTR(output.get())->size()[1] != in->size()[1] / poolsize_v_) {
        cleanup();
      }
    }

    if (output == nullptr) {
      // Check that the width and height is a multiple of the poolsize
      assert(in->size()[0] % poolsize_u_ == 0 && 
             in->size()[1] % poolsize_v_ == 0);

      std::unique_ptr<uint32_t[]> out_size(new uint32_t[in->dim()]);
      out_size[0] = in->size()[0] / poolsize_u_;
      out_size[1] = in->size()[1] / poolsize_v_;
      for (uint32_t i = 2; i < in->dim(); i++) {
        out_size[i] = in->size()[i];
      }

      output.reset(new Tensor<float>(in->dim(), out_size.get()));
      input_cpu_.reset(new float[in->nelems()]);
      output_cpu_.reset(new float[TO_TENSOR_PTR(output.get())->nelems()]);
    }

    uint32_t n_threads = 1;
    if (in->dim() > 2) {
      n_threads = TO_TENSOR_PTR(output.get())->size()[2];
    }
    if (thread_cbs_.size() != n_threads) {
      thread_cbs_.empty();
      for (uint32_t f = 0; f < n_threads; f++) {
        thread_cbs_.push_back(std::unique_ptr<jcl::threading::Callback<void>>(
          MakeCallableMany(&SpatialLPPooling::forwardPropThread, this, f)));
      }
    }
  }

  void SpatialLPPooling::forwardProp(std::shared_ptr<TorchData> input) { 
    init(input);
    Tensor<float>*in = TO_TENSOR_PTR(input.get());
    in->getData(input_cpu_.get());
    cur_in_w = in->size()[0];
    cur_in_h = in->size()[1];
    threads_finished_ = 0;
    for (uint32_t i = 0; i < thread_cbs_.size(); i++) {
      tp_->addTask(thread_cbs_[i].get());
    } 

    // Wait for all threads to finish
    std::unique_lock<std::mutex> ul(thread_update_lock_);  // Get lock
    while (threads_finished_ != static_cast<int32_t>(thread_cbs_.size())) {
      not_finished_.wait(ul);
    }
    ul.unlock();  // Release lock
    TO_TENSOR_PTR(output.get())->setData(output_cpu_.get());
  }

  void SpatialLPPooling::forwardPropThread(const uint32_t outf) {
    const uint32_t out_w = TO_TENSOR_PTR(output.get())->size()[0];
    const uint32_t out_h = TO_TENSOR_PTR(output.get())->size()[1];
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

  std::unique_ptr<TorchStage> SpatialLPPooling::loadFromFile(std::ifstream& file) {
    int filt_width, filt_height, pnorm;
    file.read((char*)(&filt_width), sizeof(filt_width));
    file.read((char*)(&filt_height), sizeof(filt_height));
    file.read((char*)(&pnorm), sizeof(pnorm));
    return std::unique_ptr<TorchStage>(
      new SpatialLPPooling((float)pnorm, filt_height, filt_width));
  }

}  // namespace jtorch
