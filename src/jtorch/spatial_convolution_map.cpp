#include "jtorch/spatial_convolution_map.h"
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

  SpatialConvolutionMap::SpatialConvolutionMap(const int32_t feats_in, 
    const int32_t feats_out, const int32_t fan_in, const int32_t filt_height, 
    const int32_t filt_width) 
    : TorchStage() {
    filt_width_ = filt_width;
    filt_height_ = filt_height;
    feats_in_ = feats_in;
    feats_out_ = feats_out;
    fan_in_ = fan_in;

    output = NULL;
    thread_cbs_ = NULL;
    output_cpu_ = NULL;
    input_cpu_ = NULL;

    weights = new float*[feats_out_ * fan_in_];
    for (int32_t i = 0; i < feats_out_ * fan_in_; i++) {
      weights[i] = new float[filt_width_ * filt_height_];
    }
    conn_table = new int16_t*[feats_out_];
    for (int32_t i = 0; i < feats_out_; i++) {
      conn_table[i] = new int16_t[fan_in_ * 2];
    }
    biases = new float[feats_out_];

    tp_ = new ThreadPool(JTIL_SPATIAL_CONVOLUTION_MAP_NTHREADS);

    std::cout << "WARNING: SPATIALCONVOLUTIONMAP IS SLOW.  All computation is";
    std::cout << " done on the CPU and incurs large transfer penalties!";
    std::cout << std::endl;
  }

  SpatialConvolutionMap::~SpatialConvolutionMap() {
    tp_->stop();
    SAFE_DELETE(tp_);
    SAFE_DELETE(output);
    SAFE_DELETE_ARR(output_cpu_);
    SAFE_DELETE_ARR(input_cpu_);
    SAFE_DELETE(thread_cbs_);
    for (int32_t i = 0; i < feats_out_ * fan_in_; i++) {
      SAFE_DELETE_ARR(weights[i]);
    }
    SAFE_DELETE(weights);
    for (int32_t i = 0; i < feats_out_; i++) {
      SAFE_DELETE_ARR(conn_table[i]);
    }
    SAFE_DELETE(conn_table);
    SAFE_DELETE_ARR(biases);
  }

  void SpatialConvolutionMap::init(TorchData& input, 
    jtil::threading::ThreadPool& tp)  {
    if (input.type() != TorchDataType::TENSOR_DATA) {
      throw std::wruntime_error("SpatialConvolutionMap::init() - "
        "FloatTensor expected!");
    }
    Tensor<float>& in = (Tensor<float>&)input;
    if (in.dim()[2] != feats_in_) {
      throw std::wruntime_error("SpatialConvolutionMap::init() - ERROR: "
        "incorrect number of input features!");
    }
    if (output != NULL) {
      Int3 out_dim(in.dim());
      out_dim[0] = out_dim[0] - filt_width_ + 1;
      out_dim[1] = out_dim[1] - filt_height_ + 1;
      out_dim[2] = feats_out_;
      if (!Int3::equal(out_dim, ((Tensor<float>*)output)->dim())) {
        // Input dimension has changed!
        SAFE_DELETE(output);
        SAFE_DELETE_ARR(output_cpu_);
        SAFE_DELETE_ARR(input_cpu_);
        SAFE_DELETE(thread_cbs_);
      }
    }
    if (output == NULL) {
      Int3 out_dim(in.dim());
      out_dim[0] = out_dim[0] - filt_width_ + 1;
      out_dim[1] = out_dim[1] - filt_height_ + 1;
      out_dim[2] = feats_out_;
      output = new Tensor<float>(out_dim);
      input_cpu_ = new float[input.dataSize()];
      output_cpu_ = new float[output->dataSize()];
    }
    if (thread_cbs_ == NULL) {
      int32_t n_feats = feats_out_;
      int32_t n_threads = n_feats;
      thread_cbs_ = new VectorManaged<Callback<void>*>(n_threads);
      for (int32_t dim2 = 0; dim2 < n_feats; dim2++) {
        thread_cbs_->pushBack(MakeCallableMany(
          &SpatialConvolutionMap::forwardPropThread, 
          this, dim2));
      }
    }
  }

  void SpatialConvolutionMap::forwardProp(TorchData& input) { 
    init(input, *tp_);
    Tensor<float>& in = (Tensor<float>&)input;
    Tensor<float>* out = (Tensor<float>*)output;
    in.getData(input_cpu_);  // Expensive O(n) copy from the GPU
    const int32_t n_banks = 1;  // No longer using 4D data
    const uint32_t in_bank_size = in.dim()[0] * in.dim()[1] * in.dim()[2];
    const uint32_t out_bank_size = out->dim()[0] * out->dim()[1] * out->dim()[2];
    for (int32_t bank = 0; bank < n_banks; bank++) {
      cur_input_ = &input_cpu_[bank * in_bank_size];
      cur_output_ = &output_cpu_[bank * out_bank_size];
      cur_input_width_ = in.dim()[0];
      cur_input_height_ = in.dim()[1];

      threads_finished_ = 0;
      for (int32_t i = 0; i < feats_out_; i++) {
        tp_->addTask((*thread_cbs_)[i]);
      } 

      // Wait for all threads to finish
      std::unique_lock<std::mutex> ul(thread_update_lock_);  // Get lock
      while (threads_finished_ != feats_out_) {
        not_finished_.wait(ul);
      }
      ul.unlock();  // Release lock
    }
    // Now copy the results up to the GPU
    out->setData(output_cpu_);
  }

  void SpatialConvolutionMap::forwardPropThread(const int32_t outf) {
    const int32_t out_w = ((Tensor<float>*)output)->dim()[0];
    const int32_t out_h = ((Tensor<float>*)output)->dim()[1];
    const int32_t out_dim = out_w * out_h;
    const int32_t in_dim = cur_input_width_ * cur_input_height_;

    // Initialize the output array to the convolution bias:
    // http://www.torch.ch/manual/nn/index#spatialconvolution
    // Set the output layer to the current bias
    for (int32_t uv = outf * out_dim; uv < ((outf+1) * out_dim); uv++) {
      cur_output_[uv] = biases[outf];
    }

    // Now iterate through the connection table:
    for (int32_t inf = 0; inf < fan_in_; inf++) {
      int32_t inf_index = (int32_t)conn_table[outf][inf * 2];
      int32_t weight_index = (int32_t)conn_table[outf][inf * 2 + 1];
      float* cur_filt = weights[weight_index];

      // for each output pixel, perform the convolution over the input
      for (int32_t outv = 0; outv < out_h; outv++) {
        for (int32_t outu = 0; outu < out_w; outu++) {
          // Now perform the convolution of the inputs
          for (int32_t filtv = 0; filtv < filt_height_; filtv++) {
            for (int32_t filtu = 0; filtu < filt_width_; filtu++) {
              int32_t inu = outu + filtu;
              int32_t inv = outv + filtv;
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

  TorchStage* SpatialConvolutionMap::loadFromFile(std::ifstream& file) {
    int32_t filt_width, filt_height, n_input_features, n_output_features;
    int32_t filt_fan_in;
    file.read((char*)(&filt_width), sizeof(filt_width));
    file.read((char*)(&filt_height), sizeof(filt_height));
    file.read((char*)(&n_input_features), sizeof(n_input_features));
    file.read((char*)(&n_output_features), sizeof(n_output_features));
    file.read((char*)(&filt_fan_in), sizeof(filt_fan_in));

    SpatialConvolutionMap* ret = new SpatialConvolutionMap(n_input_features,
      n_output_features, filt_fan_in, filt_height, filt_width);

    int32_t filt_dim = filt_width * filt_height;
    for (int32_t i = 0; i < n_output_features * filt_fan_in; i++) {
      file.read((char*)(ret->weights[i]), sizeof(ret->weights[i][0]) * filt_dim);
    }

    for (int32_t i = 0; i < n_output_features; i++) {
      file.read((char*)(ret->conn_table[i]), 
        sizeof(ret->conn_table[i][0]) * filt_fan_in * 2);
    }

    file.read((char*)(ret->biases), sizeof(ret->biases[0]) * n_output_features);
    return ret;
  }

}  // namespace jtorch