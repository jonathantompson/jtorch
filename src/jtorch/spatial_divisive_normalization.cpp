#include "jtorch/spatial_divisive_normalization.h"
#include "jtorch/tensor.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"
#include "jcl/data_str/vector_managed.h"

#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }

using namespace jcl::threading;
using namespace jcl::math;
using namespace jcl::data_str;

namespace jtorch {

  // kernel1d default is either TorchStage::gaussian1D<float>(n) or just a
  // vector of 1 values.
  SpatialDivisiveNormalization::SpatialDivisiveNormalization(
    const Tensor<float>& kernel1d, const float threshold) : TorchStage() {
    if (kernel1d.dataSize() % 2 == 0 || kernel1d.dim()[1] != 1 ||
      kernel1d.dim()[2] != 1) {
      throw std::runtime_error("SpatialDivisiveNormalization() - ERROR: "
        "Averaging kernel must be 1D and have odd size!");
    }

    kernel1d_ = kernel1d.copy();  // Normalization is input size dependant
    kernel1d_norm_ = NULL;

    output = NULL;
    std_coef_ = NULL;
    std_pass1_ = NULL;
    std_pass2_ = NULL;
    std_ = NULL;

    threshold_ = threshold;
  }

  SpatialDivisiveNormalization::~SpatialDivisiveNormalization() {
    SAFE_DELETE(output);
    SAFE_DELETE(kernel1d_);
    SAFE_DELETE(kernel1d_norm_);
    SAFE_DELETE(std_coef_);
    SAFE_DELETE(std_pass1_);
    SAFE_DELETE(std_pass2_);
    SAFE_DELETE(std_);
  }

  void SpatialDivisiveNormalization::init(TorchData& input)  {
    if (input.type() != TorchDataType::TENSOR_DATA) {
      throw std::runtime_error("SpatialDivisiveNormalization::init() - "
        "FloatTensor expected!");
    }
    Tensor<float>& in = (Tensor<float>&)input;
    if (output != NULL) {
      if (!Int3::equal(in.dim(), ((Tensor<float>*)output)->dim())) {
        // Input dimension has changed!
        SAFE_DELETE(output);
        SAFE_DELETE(kernel1d_);
        SAFE_DELETE(kernel1d_norm_);
        SAFE_DELETE(std_coef_);
        SAFE_DELETE(std_pass1_);
        SAFE_DELETE(std_pass2_);
        SAFE_DELETE(std_);
      }
    }
    if (output == NULL) {
      output = new Tensor<float>(in.dim());
      std_pass1_ = new Tensor<float>(in.dim());
      std_pass2_ = new Tensor<float>(in.dim());

      //cl_context->getOptimalLocalWorkgroupSizes(deviceid, 
      //  ((Tensor<float>*)output)->dim(), local_worgroup_size_3d);
    }
    if (kernel1d_norm_ == NULL) {
      kernel1d_norm_ = kernel1d_->copy();
      float* kernel1d_norm_cpu = new float[kernel1d_norm_->dataSize()];
      kernel1d_norm_->getData(kernel1d_norm_cpu);
      // Now normalize the kernel!
      const float n_feats = (float)in.dim()[2];
      float sum = 0.0f;
      for (int32_t i = 0; i < kernel1d_norm_->dim()[0]; i++) {
        sum += kernel1d_norm_cpu[i];
      }
      for (int32_t i = 0; i < kernel1d_norm_->dim()[0]; i++) {
        kernel1d_norm_cpu[i] /= (sum * sqrtf(n_feats));
      }
      kernel1d_norm_->setData(kernel1d_norm_cpu);
      delete[] kernel1d_norm_cpu;
    }
    if (std_coef_ == NULL) {
      Int3 std_coeff_dim(((Tensor<float>*)output)->dim());
      std_coeff_dim[2] = 1;  // This tensor is only 2D
      std_coef_ = new Tensor<float>(std_coeff_dim);

      float* std_coef_cpu = new float[std_coef_->dataSize()];
      float* kernel1d_norm_cpu = new float[kernel1d_norm_->dataSize()];
      kernel1d_norm_->getData(kernel1d_norm_cpu);

      // Filter an image of all 1 values to create the normalization constants
      // See norm_test.lua for proof that this works as well as:
      // https://github.com/andresy/torch/blob/master/extra/nn/SpatialSubtractiveNormalization.lua
      // The filter is seperable, but we'll just do the dumb 2D version since
      // we only do this once on startup.  --> O(n * m)
      int32_t kernel1d_size = kernel1d_->dim()[0];
      int32_t filt_rad = (kernel1d_size - 1) / 2;
      int32_t n_feats = ((Tensor<float>*)output)->dim()[2];
      int32_t height = ((Tensor<float>*)output)->dim()[1];
      int32_t width = ((Tensor<float>*)output)->dim()[0];
      for (int32_t v = 0; v < height; v++) {
        for (int32_t u = 0; u < width; u++) {
          std_coef_cpu[v * width + u] = 0.0f;
          for (int32_t v_filt = -filt_rad; v_filt <= filt_rad; v_filt++) {
            for (int32_t u_filt = -filt_rad; u_filt <= filt_rad; u_filt++) {
              int32_t u_in = u + u_filt;
              int32_t v_in = v + v_filt;
              if (u_in >= 0 && u_in < width && v_in >= 0 && v_in < height) {
                // Pixel is inside --> We'll effectively clamp zeros elsewhere.
                std_coef_cpu[v * width + u] += 
                  (kernel1d_norm_cpu[v_filt + filt_rad] * 
                   kernel1d_norm_cpu[u_filt + filt_rad]);
              }
            }
          }
          std_coef_cpu[v * width + u] /= n_feats;
        }
      }
      std_coef_->setData(std_coef_cpu);
      delete[] std_coef_cpu;
      delete[] kernel1d_norm_cpu;
    }
    if (std_ == NULL) {
      Int3 std_coeff_dim(((Tensor<float>*)output)->dim());
      std_coeff_dim[2] = 1;  // This tensor is only 2D
      std_ = new Tensor<float>(std_coeff_dim);

      //cl_context->getOptimalLocalWorkgroupSizes(deviceid, std_->dim(), 
      //  local_worgroup_size_2d);
    }
  }

  void SpatialDivisiveNormalization::forwardProp(TorchData& input) { 
    init(input);
    int32_t filt_rad = (kernel1d_norm_->dim()[0] - 1) / 2;
    
    // Perform horizontal filter pass
    Tensor<float>& in = (Tensor<float>&)input;
    Tensor<float>* out = (Tensor<float>*)output;
    std::string kernel = jtorch::jtorch_path + "kernels/spatial_divisive_normalization.cl";
    cl_context->useKernel(kernel.c_str(), "SpatialDivisiveNormalizationHoriz");
    cl_context->setArg(0, in.data());
    cl_context->setArg(1, std_pass1_->data());
    cl_context->setArg(2, kernel1d_norm_->data());
    cl_context->setArg(3, filt_rad);
    cl_context->runKernel3D(jtorch::deviceid, std_pass1_->dim(), false);

    // Perform vertical filter pass
    cl_context->useKernel(kernel.c_str(), "SpatialDivisiveNormalizationVert");
    cl_context->setArg(0, std_pass1_->data());
    cl_context->setArg(1, std_pass2_->data());
    cl_context->setArg(2, kernel1d_norm_->data());
    cl_context->setArg(3, filt_rad);
    cl_context->runKernel3D(jtorch::deviceid, std_pass2_->dim(), false);

    // Perform accumulation and division pass
    cl_context->useKernel(kernel.c_str(), "SpatialDivisiveNormalizationAccumDiv");
    cl_context->setArg(0, std_pass2_->data());
    cl_context->setArg(1, std_->data());
    cl_context->setArg(2, std_coef_->data());
    cl_context->setArg(3, out->dim()[2]);
    cl_context->setArg(4, threshold_);
    cl_context->runKernel3D(jtorch::deviceid, std_->dim(), false);

    // Perform normalization pass
    cl_context->useKernel(kernel.c_str(), "SpatialDivisiveNormalization");
    cl_context->setArg(0, in.data());
    cl_context->setArg(1, out->data());
    cl_context->setArg(2, std_->data());
    cl_context->runKernel3D(jtorch::deviceid, out->dim(), false);
  }

  TorchStage* SpatialDivisiveNormalization::loadFromFile(std::ifstream& file) {
    // This whole thing is a little wasteful.  I copy to GPU here, and then
    // I copy it back down in the constructor anyway...  But it's good enough
    // for now.
    int32_t kernel_size;
    file.read((char*)(&kernel_size), sizeof(kernel_size));
    Tensor<float>* kernel = new Tensor<float>(kernel_size);
    float* kernel_cpu = new float[kernel_size];
    file.read((char*)(kernel_cpu), kernel_size * sizeof(*kernel_cpu));
    float threshold;
    file.read((char*)(&threshold), sizeof(threshold));
    kernel->setData(kernel_cpu);
    TorchStage* ret = new SpatialDivisiveNormalization(*kernel, threshold);
    delete kernel;
    delete[] kernel_cpu;
    return ret;
  }

}  // namespace jtorch