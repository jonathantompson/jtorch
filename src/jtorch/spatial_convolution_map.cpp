#include "jtorch/spatial_convolution_map.h"
#include "jtorch/tensor.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"
#include "jcl/data_str/vector_managed.h"

using namespace jcl::threading;
using namespace jcl::math;
using namespace jcl::data_str;

namespace jtorch {

  SpatialConvolutionMap::SpatialConvolutionMap(const uint32_t feats_in, 
    const uint32_t feats_out, const uint32_t fan_in, 
    const uint32_t filt_height, const uint32_t filt_width) 
    : TorchStage() {
    filt_width_ = filt_width;
    filt_height_ = filt_height;
    feats_in_ = feats_in;
    feats_out_ = feats_out;
    fan_in_ = fan_in;

    output = nullptr;
    output_cpu_.reset(nullptr);
    input_cpu_.reset(nullptr);

    for (uint32_t i = 0; i < feats_out_ * fan_in_; i++) {
      weights.push_back(std::unique_ptr<float[]>(new float[filt_width_ * filt_height_]));
    }
    for (uint32_t i = 0; i < feats_out_; i++) {
      conn_table.push_back(std::unique_ptr<int16_t[]>(new int16_t[fan_in_ * 2]));
    }
    biases.reset(new float[feats_out_]);

    tp_.reset(new ThreadPool(JTIL_SPATIAL_CONVOLUTION_MAP_NTHREADS));

    std::cout << "WARNING: SPATIALCONVOLUTIONMAP IS SLOW." << std::endl;
    std::cout << "--> ALL COMPUTATION IS DONE ON THE CPU!" << std::endl;
  }

  SpatialConvolutionMap::~SpatialConvolutionMap() {
    tp_->stop();
  }

  void SpatialConvolutionMap::init(std::shared_ptr<TorchData> input)  {
    assert(input->type() == TorchDataType::TENSOR_DATA);
    
    Tensor<float>* in = TO_TENSOR_PTR(input.get());
    assert(in->dim() == 3);
    assert(in->size()[2] == feats_in_);

    if (output != nullptr) {
      uint32_t owidth = in->size()[0] - filt_width_ + 1;
      uint32_t oheight = in->size()[1] - filt_height_ + 1;
      const uint32_t* out_size = TO_TENSOR_PTR(output.get())->size();
      if (out_size[0] != owidth || out_size[1] != oheight || 
          out_size[2] != feats_out_) {
        // Input dimension has changed!
        output = nullptr;
        output_cpu_.reset(nullptr);
        input_cpu_.reset(nullptr);
        thread_cbs_.clear();
      }
    }
    if (output == nullptr) {
      uint32_t out_dim[3];
      out_dim[0] = in->size()[0] - filt_width_ + 1;
      out_dim[1] = in->size()[1] - filt_height_ + 1;
      out_dim[2] = feats_out_;
      output.reset(new Tensor<float>(3, out_dim));
      input_cpu_.reset(new float[in->nelems()]);
      output_cpu_.reset(new float[TO_TENSOR_PTR(output.get())->nelems()]);
    }
    uint32_t n_feats = feats_out_;
    uint32_t n_threads = n_feats;
    if (thread_cbs_.size() != n_threads) {
      thread_cbs_.clear();
      for (uint32_t dim2 = 0; dim2 < n_feats; dim2++) {
        thread_cbs_.push_back(std::unique_ptr<jcl::threading::Callback<void>>(
          MakeCallableMany(&SpatialConvolutionMap::forwardPropThread, 
                           this, dim2)));
      }
    }
  }

  void SpatialConvolutionMap::forwardProp(std::shared_ptr<TorchData> input) { 
    init(input);
    Tensor<float>* in = TO_TENSOR_PTR(input.get());
    Tensor<float>* out = TO_TENSOR_PTR(output.get());
    in->getData(input_cpu_.get());  // Expensive O(n) copy from the GPU
    const int32_t n_banks = 1;  // No longer using 4D data
    const uint32_t in_bank_size = in->size()[0] * in->size()[1] * in->size()[2];
    const uint32_t out_bank_size = out->size()[0] * out->size()[1] * out->size()[2];
    for (int32_t bank = 0; bank < n_banks; bank++) {
      cur_input_ = &input_cpu_[bank * in_bank_size];
      cur_output_ = &output_cpu_[bank * out_bank_size];
      cur_input_width_ = in->size()[0];
      cur_input_height_ = in->size()[1];

      threads_finished_ = 0;
      for (uint32_t i = 0; i < feats_out_; i++) {
        tp_->addTask(thread_cbs_[i].get());
      } 

      // Wait for all threads to finish
      std::unique_lock<std::mutex> ul(thread_update_lock_);  // Get lock
      while (threads_finished_ != feats_out_) {
        not_finished_.wait(ul);
      }
      ul.unlock();  // Release lock
    }
    // Now copy the results up to the GPU
    out->setData(output_cpu_.get());
  }

  void SpatialConvolutionMap::forwardPropThread(const uint32_t outf) {
    const uint32_t out_w = TO_TENSOR_PTR(output.get())->size()[0];
    const uint32_t out_h = TO_TENSOR_PTR(output.get())->size()[1];
    const uint32_t out_dim = out_w * out_h;
    const uint32_t in_dim = cur_input_width_ * cur_input_height_;

    // Initialize the output array to the convolution bias:
    // http://www.torch.ch/manual/nn/index#spatialconvolution
    // Set the output layer to the current bias
    for (uint32_t uv = outf * out_dim; uv < ((outf+1) * out_dim); uv++) {
      cur_output_[uv] = biases[outf];
    }

    // Now iterate through the connection table:
    for (uint32_t inf = 0; inf < fan_in_; inf++) {
      uint32_t inf_index = (int32_t)conn_table[outf][inf * 2];
      uint32_t weight_index = (int32_t)conn_table[outf][inf * 2 + 1];
      float* cur_filt = weights[weight_index].get();

      // for each output pixel, perform the convolution over the input
      for (uint32_t outv = 0; outv < out_h; outv++) {
        for (uint32_t outu = 0; outu < out_w; outu++) {
          // Now perform the convolution of the inputs
          for (uint32_t filtv = 0; filtv < filt_height_; filtv++) {
            for (uint32_t filtu = 0; filtu < filt_width_; filtu++) {
              uint32_t inu = outu + filtu;
              uint32_t inv = outv + filtv;
              cur_output_[outf * out_dim + outv * out_w + outu] +=
                (cur_filt[filtv * filt_width_ + filtu] *
                cur_input_[inf_index * in_dim + inv * cur_input_width_ + inu]);
            }
          }
        }
      }
      // Convolution finished for this input feature
    }
    std::unique_lock<std::mutex> ul(thread_update_lock_);
    threads_finished_++;
    not_finished_.notify_all();  // Signify that all threads might have finished
    ul.unlock();
  }

  std::unique_ptr<TorchStage> SpatialConvolutionMap::loadFromFile(std::ifstream& file) {
    int32_t filt_width, filt_height, n_input_features, n_output_features;
    int32_t filt_fan_in;
    file.read((char*)(&filt_width), sizeof(filt_width));
    file.read((char*)(&filt_height), sizeof(filt_height));
    file.read((char*)(&n_input_features), sizeof(n_input_features));
    file.read((char*)(&n_output_features), sizeof(n_output_features));
    file.read((char*)(&filt_fan_in), sizeof(filt_fan_in));

    std::unique_ptr<SpatialConvolutionMap> ret(
      new SpatialConvolutionMap(n_input_features, n_output_features, 
                                filt_fan_in, filt_height, filt_width));

    int32_t filt_dim = filt_width * filt_height;
    for (int32_t i = 0; i < n_output_features * filt_fan_in; i++) {
      file.read((char*)(ret->weights[i].get()), sizeof(ret->weights[i][0]) * filt_dim);
    }

    for (int32_t i = 0; i < n_output_features; i++) {
      file.read((char*)(ret->conn_table[i].get()), 
        sizeof(ret->conn_table[i][0]) * filt_fan_in * 2);
    }

    file.read((char*)(ret->biases.get()), sizeof(ret->biases[0]) * n_output_features);
    return std::unique_ptr<TorchStage>(std::move(ret));
  }

}  // namespace jtorch