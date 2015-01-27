#include "jtorch/spatial_subtractive_normalization.h"
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
  SpatialSubtractiveNormalization::SpatialSubtractiveNormalization(
    const Tensor<float>& kernel) : TorchStage() {
    if (kernel.dim()[0] % 2 == 0 || kernel.dim()[1] % 2 == 0 ||
        kernel.dim()[2] != 1) {
      throw std::runtime_error("SpatialSubtractiveNormalization() - ERROR: "
        "Averaging kernel must have odd size!");
    }

    // Clone and normalize the input kernel
    kernel_ = Tensor<float>::clone(kernel);
    float sum = Tensor<float>::slowSum(*kernel_);
    Tensor<float>::div(*kernel_, sum);

    output = NULL;
    mean_coef_ = NULL;
    mean_pass1_ = NULL;
    mean_pass2_ = NULL;
    mean_ = NULL;
  }

  SpatialSubtractiveNormalization::~SpatialSubtractiveNormalization() {
    SAFE_DELETE(output);
    SAFE_DELETE(kernel_);
    SAFE_DELETE(mean_coef_);
    SAFE_DELETE(mean_pass1_);
    SAFE_DELETE(mean_pass2_);
    SAFE_DELETE(mean_);
  }

  void SpatialSubtractiveNormalization::init(TorchData& input)  {
    if (input.type() != TorchDataType::TENSOR_DATA) {
      throw std::runtime_error("SpatialSubtractiveNormalization::init() - "
        "FloatTensor expected!");
    }
    Tensor<float>& in = (Tensor<float>&)input;
    if (output != NULL) {
      if (!Int3::equal(in.dim(), ((Tensor<float>*)output)->dim())) {
        // Input dimension has changed!
        SAFE_DELETE(output);
        SAFE_DELETE(mean_coef_);
        SAFE_DELETE(mean_pass1_);
        SAFE_DELETE(mean_pass2_);
        SAFE_DELETE(mean_);
      }
    }
    if (output == NULL) {
      output = new Tensor<float>(in.dim());
      mean_pass1_ = new Tensor<float>(in.dim());
      mean_pass2_ = new Tensor<float>(in.dim());

      //cl_context->getOptimalLocalWorkgroupSizes(deviceid, 
      //  ((Tensor<float>*)output)->dim(), local_worgroup_size_3d);
    }
    if (mean_coef_ == NULL) {
      Int3 mean_coeff_dim(((Tensor<float>*)output)->dim());
      mean_coeff_dim[2] = 1;  // This tensor is only 2D
      mean_coef_ = new Tensor<float>(mean_coeff_dim);

      float* mean_coef_cpu = new float[mean_coef_->dataSize()];
      float* kernel_cpu = new float[kernel_->dataSize()];
      kernel_->getData(kernel_cpu);
      bool onedim_kernel = kernel_->dim()[1] == 1;

      // Filter an image of all 1 values to create the normalization constants
      // See norm_test.lua for proof that this works as well as:
      // https://github.com/andresy/torch/blob/master/extra/nn/SpatialSubtractiveNormalization.lua
      int32_t n_feats = ((Tensor<float>*)output)->dim()[2];
      int32_t height = ((Tensor<float>*)output)->dim()[1];
      int32_t width = ((Tensor<float>*)output)->dim()[0];
      if (onedim_kernel) {
        // 1D case - The filter is seperable, but we'll just do the dumb 2D 
        // version since we only do this once on startup.  --> O(n * m)
        int32_t kernel_size = kernel_->dim()[0];
        int32_t filt_rad = (kernel_size - 1) / 2;
        for (int32_t v = 0; v < height; v++) {
          for (int32_t u = 0; u < width; u++) {
            float tmp = 0.0f;
            for (int32_t v_filt = -filt_rad; v_filt <= filt_rad; v_filt++) {
              for (int32_t u_filt = -filt_rad; u_filt <= filt_rad; u_filt++) {
                int32_t u_in = u + u_filt;
                int32_t v_in = v + v_filt;
                if (u_in >= 0 && u_in < width && v_in >= 0 && v_in < height) {
                  // Pixel is inside --> We'll effectively clamp zeros elsewhere.
                  tmp += 
                    (kernel_cpu[v_filt + filt_rad] * kernel_cpu[u_filt + filt_rad]);
                }
              }
            }
            mean_coef_cpu[v * width + u] = tmp / n_feats;
          }
        }
      } else {
        // 2D case
        int32_t kernel_size_u = kernel_->dim()[0];
        int32_t kernel_size_v = kernel_->dim()[1];
        int32_t filt_rad_u = (kernel_size_u - 1) / 2;
        int32_t filt_rad_v = (kernel_size_v - 1) / 2;
        for (int32_t v = 0; v < height; v++) {
          for (int32_t u = 0; u < width; u++) {
            float tmp = 0.0f;
            for (int32_t v_filt = -filt_rad_v; v_filt <= filt_rad_v; v_filt++) {
              for (int32_t u_filt = -filt_rad_u; u_filt <= filt_rad_u; u_filt++) {
                int32_t u_in = u + u_filt;
                int32_t v_in = v + v_filt;
                if (u_in >= 0 && u_in < width && v_in >= 0 && v_in < height) {
                  // Pixel is inside --> We'll effectively clamp zeros elsewhere.
                  tmp += 
                    kernel_cpu[(v_filt + filt_rad_v) * kernel_size_u + (u_filt + filt_rad_u)];
                }
              }
            }
            mean_coef_cpu[v * width + u] = tmp / n_feats;
          }
        }
      }
      mean_coef_->setData(mean_coef_cpu);
      delete[] mean_coef_cpu;
      delete[] kernel_cpu;
    }
    if (mean_ == NULL) {
      Int3 mean_coeff_dim(((Tensor<float>*)output)->dim());
      mean_coeff_dim[2] = 1;  // This tensor is only 2D
      mean_ = new Tensor<float>(mean_coeff_dim);

      //cl_context->getOptimalLocalWorkgroupSizes(deviceid, mean_->dim(), 
      //  local_worgroup_size_2d);
    }
  }

  void SpatialSubtractiveNormalization::forwardProp(TorchData& input) { 
    init(input);
    bool onedim_kernel = kernel_->dim()[1] == 1;

    Tensor<float>& in = (Tensor<float>&)input;
    Tensor<float>* out = (Tensor<float>*)output;
    std::string kernel = jtorch::jtorch_path + "kernels/spatial_subtractive_normalization.cl";

    if (onedim_kernel) {
      int32_t filt_rad = (kernel_->dim()[0] - 1) / 2;
    
      // Perform horizontal filter pass
      cl_context->useKernel(kernel.c_str(), "SpatialSubtractiveNormalizationHoriz");
      cl_context->setArg(0, in.data());
      cl_context->setArg(1, mean_pass1_->data());
      cl_context->setArg(2, kernel_->data());
      cl_context->setArg(3, filt_rad);
      cl_context->runKernel3D(jtorch::deviceid, mean_pass1_->dim(), false);

      // Perform vertical filter pass
      cl_context->useKernel(kernel.c_str(), "SpatialSubtractiveNormalizationVert");
      cl_context->setArg(0, mean_pass1_->data());
      cl_context->setArg(1, mean_pass2_->data());
      cl_context->setArg(2, kernel_->data());
      cl_context->setArg(3, filt_rad);
      cl_context->runKernel3D(jtorch::deviceid, mean_pass2_->dim(), false);
    } else {
      int32_t filt_rad_u = (kernel_->dim()[0] - 1) / 2;
      int32_t filt_rad_v = (kernel_->dim()[1] - 1) / 2;
    
      // Perform horizontal filter pass
      cl_context->useKernel(kernel.c_str(), "SpatialSubtractiveNormalization2D");
      cl_context->setArg(0, in.data());
      cl_context->setArg(1, mean_pass2_->data());
      cl_context->setArg(2, kernel_->data());
      cl_context->setArg(3, filt_rad_u);
      cl_context->setArg(4, filt_rad_v);
      cl_context->runKernel3D(jtorch::deviceid, mean_pass2_->dim(), false);
    }

    // Perform accumulation and division pass
    cl_context->useKernel(kernel.c_str(), "SpatialSubtractiveNormalizationAccumDiv");
    cl_context->setArg(0, mean_pass2_->data());
    cl_context->setArg(1, mean_->data());
    cl_context->setArg(2, mean_coef_->data());
    cl_context->setArg(3, out->dim()[2]);
    cl_context->runKernel3D(jtorch::deviceid, mean_->dim(), false);

    // Perform normalization pass
    cl_context->useKernel(kernel.c_str(), "SpatialSubtractiveNormalization");
    cl_context->setArg(0, in.data());
    cl_context->setArg(1, out->data());
    cl_context->setArg(2, mean_->data());
    cl_context->runKernel3D(jtorch::deviceid, out->dim(), false);
  }

  TorchStage* SpatialSubtractiveNormalization::loadFromFile(std::ifstream& file) {
    // This whole thing is a little wasteful.  I copy to GPU here, and then
    // I copy it back down in the constructor anyway...  But it's good enough
    // for now.
    int32_t kernel_size_2, kernel_size_1;  // kernel_size_1 is the inner dim
    file.read((char*)(&kernel_size_1), sizeof(kernel_size_1));
    file.read((char*)(&kernel_size_2), sizeof(kernel_size_2));
    Tensor<float>* kernel;
    if (kernel_size_2 > 1) {
      // The kernel is 2D
      kernel = new Tensor<float>(Int2(kernel_size_1, kernel_size_2));
    } else {
      kernel = new Tensor<float>(kernel_size_1);
    }
    float* kernel_cpu = new float[kernel->dataSize()];
    file.read((char*)(kernel_cpu), kernel->dataSize() * sizeof(*kernel_cpu));
    kernel->setData(kernel_cpu);
    TorchStage* ret = new SpatialSubtractiveNormalization(*kernel);
    delete kernel;
    delete[] kernel_cpu;
    return ret;
  }

}  // namespace jtorch