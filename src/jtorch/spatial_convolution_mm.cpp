#include "jtorch/spatial_convolution_mm.h"
#include "jtorch/tensor.h"
#include "jtorch/jtorch.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"
#include "jcl/data_str/vector_managed.h"
#include "clBLAS.h"
#include "jcl/cl_include.h"

#define SAFE_DELETE(x) if (x != nullptr) { delete x; x = nullptr; }
#define SAFE_DELETE_ARR(x) if (x != nullptr) { delete[] x; x = nullptr; }

using namespace jcl::threading;
using namespace jcl::math;
using namespace jcl::data_str;
using namespace jcl;

namespace jtorch {

  // Function signatures from Torch (for easy code reuse
  void THCudaBlas_gemm(void* state, char transa, char transb, size_t m, size_t n, size_t k, float alpha, Tensor<float> *a, size_t lda, Tensor<float> *b, size_t ldb, float beta, Tensor<float> *c, size_t ldc);
  void im2col(const Tensor<float>* data_im, const int channels, const int height, const int width, const int ksize_h, const int ksize_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, Tensor<float>* data_col);

  SpatialConvolutionMM::SpatialConvolutionMM(const uint32_t feats_in, 
    const uint32_t feats_out, const uint32_t filt_height, 
    const uint32_t filt_width, const uint32_t padding) : TorchStage() {
    filt_width_ = filt_width;
    filt_height_ = filt_height;
    feats_in_ = feats_in;
    feats_out_ = feats_out;
    padding_ = padding;

    output = nullptr;
    ones_ = nullptr;
    columns_ = nullptr;

    uint32_t dim = 4;
    uint32_t size[4] = {filt_width_, filt_height_, feats_in_, feats_out_};
    weights_ = new Tensor<float>(dim, size);
    biases_ = new Tensor<float>(1, &feats_out_);
  }

  SpatialConvolutionMM::~SpatialConvolutionMM() {
    SAFE_DELETE(output);
    SAFE_DELETE(weights_);
    SAFE_DELETE(biases_);
    SAFE_DELETE(columns_);
    SAFE_DELETE(ones_);
  }

  void SpatialConvolutionMM::setWeights(const float* weights) {
    weights_->setData(weights);
  }

  void SpatialConvolutionMM::setBiases(const float* biases) {
    biases_->setData(biases);
  }

  void SpatialConvolutionMM::init(TorchData& input)  {
    if (input.type() != TorchDataType::TENSOR_DATA) {
      throw std::runtime_error("SpatialConvolutionMM::init() - "
        "FloatTensor expected!");
    }
    Tensor<float>& in = (Tensor<float>&)input;
    if (in.dim() != 3) {
      throw std::runtime_error("SpatialConvolutionMM::init() - Input not 3D!");
    }
    if (in.size()[2] != feats_in_) {
      throw std::runtime_error("SpatialConvolutionMM::init() - ERROR: "
        "incorrect number of input features!");
    }
    if (output != nullptr) {  
      uint32_t owidth = in.size()[0] - filt_width_ + 1 + 2 * padding_;
      uint32_t oheight  = in.size()[1] - filt_height_ + 1 + 2 * padding_;
      const uint32_t* out_size = TO_TENSOR_PTR(output)->size();
      if (out_size[0] != owidth || out_size[1] != oheight || 
        out_size[2] != feats_out_) {
        // Output size changed
        SAFE_DELETE(output);
        SAFE_DELETE(columns_);
        SAFE_DELETE(ones_);
      }
    }

    if (output == nullptr) {
      const uint32_t inputWidth = in.size()[0];
      const uint32_t inputHeight = in.size()[1];
      const uint32_t outputWidth = inputWidth - filt_width_ + 1 + 2 * padding_;
      const uint32_t outputHeight = inputHeight - filt_height_ + 1 + 2 * padding_;

      // Resize output
      uint32_t out_dim[3];
      out_dim[0] = outputWidth;
      out_dim[1] = outputHeight;
      out_dim[2] = feats_out_;
      output = new Tensor<float>(3, out_dim);

      // Resize temporary columns
      uint32_t columns_dim[2];
      columns_dim[0] = outputHeight*outputWidth;
      columns_dim[1] = feats_in_ * filt_width_ * filt_height_;
      columns_ = new Tensor<float>(2, columns_dim);

      // Define a buffer of ones, for bias accumulation
      // Note: this buffer can be shared with other modules, it only ever gets increased,
      // and always contains ones.
      uint32_t ones_dim[2];
      ones_dim[0] = outputWidth;
      ones_dim[1] = outputHeight;
      ones_ = new Tensor<float>(2, ones_dim);
    }
  }

  void SpatialConvolutionMM::forwardProp(TorchData& input) { 
    init(input);

    Tensor<float>* output_n = TO_TENSOR_PTR(output);
    Tensor<float>* input_n = TO_TENSOR_PTR(&input);
    void* state = nullptr;

    const uint32_t inputWidth = input_n->size()[0];
    const uint32_t inputHeight = input_n->size()[1];
    const uint32_t outputWidth  = inputWidth - filt_width_ + 1 + 2 * padding_;
    const uint32_t outputHeight = inputHeight - filt_height_ + 1 + 2 * padding_;
    const uint32_t nInputPlane = feats_in_;
    const uint32_t nOutputPlane = feats_out_;
    const uint32_t kH = filt_height_;
    const uint32_t kW = filt_width_;
    const uint32_t padding = padding_;
    const uint32_t dH = 1;
    const uint32_t dW = 1;

    // Do Bias first:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    uint32_t m_ = nOutputPlane;
    uint32_t n_ = outputHeight * outputWidth;
    uint32_t k_ = 1;
    Tensor<float>::fill(*ones_, 1);
    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THCudaBlas_gemm(
        state,
        't', 'n',
        n_, m_, k_,
        1,
        ones_, k_,
        biases_, k_,
        0,
        output_n, n_
    );

    // Extract columns:
    im2col(
        input_n,
        nInputPlane, inputHeight, inputWidth, kH, kW, padding, padding, dH, dW,
        columns_
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    // Note: 
    // 1. In torch the weight matrix is defined as:
    //    self.weight = torch.Tensor(nOutputPlane, nInputPlane*kH*kW)
    // 2. In torch the size array is backwards, so size[0] is the highest dim, ie
    //    long batchSize = input->size[0];

    // long m = weight->size[0];
    // long n = columns->size[1];
    // long k = weight->size[1];

    long m = nOutputPlane;
    long n = outputHeight * outputWidth;
    long k = nInputPlane  *kH * kW;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THCudaBlas_gemm(
        state,
        'n', 'n',
        n, m, k,
        1,
        columns_, n,
        weights_, k,
        1,
        output_n, n
    );
  }

  TorchStage* SpatialConvolutionMM::loadFromFile(std::ifstream& file) {
    int32_t filt_width, filt_height, n_input_features, n_output_features,
      padding;
    file.read((char*)(&filt_width), sizeof(filt_width));
    file.read((char*)(&filt_height), sizeof(filt_height));
    file.read((char*)(&n_input_features), sizeof(n_input_features));
    file.read((char*)(&n_output_features), sizeof(n_output_features));
    file.read((char*)(&padding), sizeof(padding));

#if defined(DEBUG) || defined(_DEBUG)
    std::cout << "\t\t(fout,fin,kh,kw,pad)=(" << n_output_features << "," << 
      n_input_features << "," << filt_height << "," << filt_width << "," << 
      padding << ")" << std::endl;
#endif

    SpatialConvolutionMM* ret = new SpatialConvolutionMM(n_input_features,
      n_output_features, filt_height, filt_width, padding);

    int32_t filt_dim = filt_width * filt_height;
    float* weights = new float[n_output_features * n_input_features * filt_dim];
    for (int32_t i = 0; i < n_output_features * n_input_features; i++) {
      float* bank = &weights[i * filt_dim];
      file.read((char*)(bank), sizeof(bank[0]) * filt_dim);
    }
    ret->setWeights(weights);
    delete[] weights;

    float* biases = new float[n_output_features];
    file.read((char*)(biases), sizeof(biases[0]) * n_output_features);
    ret->setBiases(biases);
    delete[] biases;

    return ret;
  }

  void adjustLd(char transa, char transb, size_t m, size_t n, size_t k, 
    size_t *lda, size_t *ldb, size_t *ldc) {
    int transa_ = ((transa == 't') || (transa == 'T'));
    int transb_ = ((transb == 't') || (transb == 'T'));

    if(n == 1)
      *ldc = m;

    if(transa_)
    {
      if(m == 1)
        *lda = k;
    }
    else
    {
      if(k == 1)
        *lda = m;
    }

    if(transb_)
    {
      if(k == 1)
        *ldb = n;
    }
    else
    {
      if(n == 1)
        *ldb = k;
    }
  }

  clblasTranspose convertTransToCublasOperation(char trans) {
    if (trans == 't') return clblasTrans;  // Or CUBLAS_OP_T
    else if (trans == 'n') return clblasNoTrans;  // Or CUBLAS_OP_N
    else if (trans == 'c') return clblasConjTrans;  // Or CUBLAS_OP_C
    else {
      throw std::runtime_error("trans must be one of: t, n, c");
    }
  }

  std::string getErrorString(cl_int err) {
    switch (err) {
    case clblasNotImplemented: 
      return "clblasNotImplemented: Functionality is not implemented";
    case clblasNotInitialized: 
      return "clblasNotInitialized: clblas library is not initialized yet";
    case clblasInvalidMatA:
      return "clblasInvalidMatA: Matrix A is not a valid memory object";
    case clblasInvalidMatB:
      return "clblasInvalidMatB: Matrix B is not a valid memory object";
    case clblasInvalidMatC:
      return "clblasInvalidMatC: Matrix C is not a valid memory object";
    case clblasInvalidVecX:
      return "clblasInvalidVecX: Vector X is not a valid memory object";
    case clblasInvalidVecY:
      return "clblasInvalidVecY: Vector Y is not a valid memory object";
    case clblasInvalidDim:                     
      return "clblasInvalidDim: An input dimension (M,N,K) is invalid";
    case clblasInvalidLeadDimA:                
      return "clblasInvalidLeadDimA: Leading dimension A must not be less than the size of the first dimension";
    case clblasInvalidLeadDimB:                
      return "clblasInvalidLeadDimB: Leading dimension B must not be less than the size of the second dimension";
    case clblasInvalidLeadDimC:                
      return "clblasInvalidLeadDimC: Leading dimension C must not be less than the size of the third dimension";
    case clblasInvalidIncX:                    
      return "clblasInvalidIncX: The increment for a vector X must not be 0";
    case clblasInvalidIncY:                    
      return "clblasInvalidIncY: The increment for a vector Y must not be 0";
    case clblasInsufficientMemMatA:           
      return "clblasInsufficientMemMatA: The memory object for Matrix A is too small";
    case clblasInsufficientMemMatB:            
      return "clblasInsufficientMemMatB: The memory object for Matrix B is too small";
    case clblasInsufficientMemMatC:            
      return "clblasInsufficientMemMatC: The memory object for Matrix C is too small";
    case clblasInsufficientMemVecX:            
      return "clblasInsufficientMemVecX: The memory object for Vector X is too small";
    case clblasInsufficientMemVecY:             
      return "clblasInsufficientMemVecY: The memory object for Vector Y is too small";
    default:
      return jcl::JCL::getErrorString(err);
    }
  }

  // For easy reuse of code, just redefine THCudaBlas_gemm from torch
  void THCudaBlas_gemm(void* state, char transa, char transb, size_t m, size_t n, size_t k, float alpha, Tensor<float> *a, size_t lda, Tensor<float> *b, size_t ldb, float beta, Tensor<float> *c, size_t ldc) {
    adjustLd(transa, transb, m, n, k, &lda, &ldb, &ldc);
    clblasTranspose opa = convertTransToCublasOperation(transa);
    clblasTranspose opb = convertTransToCublasOperation(transb);

    // The Torch call:
    // THCublasCheck(cublasSgemm(*state->blasState->current_handle, 
    //   opa, 
    //   opb, 
    //   i_m, 
    //   i_n, 
    //   i_k,
    //   &alpha, 
    //   a, 
    //   i_lda, 
    //   b, 
    //   i_ldb, 
    //   &beta, 
    //   c, 
    //   i_ldc));


    clblasOrder order = clblasColumnMajor;  // Not sure what this is
    cl_command_queue queue = (cl_command_queue)cl_context->queue(jtorch::deviceid);
    cl_event event = nullptr;
    cl_int err = clblasSgemm(
      order, 
      opa, 
      opb, 
      m, 
      n, 
      k, 
      alpha, 
      (cl_mem)cl_context->getCLMem(a->storage()), 
      0,  // (offA)
      lda, 
      (cl_mem)cl_context->getCLMem(b->storage()), 
      0,  // (offB)
      ldb, 
      beta, 
      (cl_mem)cl_context->getCLMem(c->storage()), 
      0,  // (offC)
      ldc,
      1, &queue, 0, nullptr, &event);

    // Non-blocking: Don't wait for events
    // err = clWaitForEvents( 1, &event );

    if (err != CL_SUCCESS) {
      std::stringstream ss;
      ss << "Error clblasSgemm failed: " << getErrorString(err);
      throw std::runtime_error(ss.str());
    }
  }

  void im2col(const Tensor<float>* data_im, const int channels,
    const int height, const int width, const int ksize_h, const int ksize_w, 
    const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
    Tensor<float>* data_col) {
      // We are going to launch channels * height_col * width_col kernels, each
      // kernel responsible for copying a single-channel grid.
      int height_col = (height + 2 * pad_h - ksize_h) / stride_h + 1;
      int width_col = (width + 2 * pad_w - ksize_w) / stride_w + 1;
      int num_kernels = channels * height_col * width_col;
      // Launch

      // The call in torch
      //  im2col_kernel <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>> (
      //    num_kernels,   // 0
      //    data_im,       // 1
      //    height,        // 2
      //    width,         // 3
      //    ksize_h,       // 4
      //    ksize_w,       // 5
      //    pad_h,         // 6
      //    pad_w,         // 7
      //    stride_h,      // 8
      //    stride_w,      // 9
      //    height_col,    // 10
      //    width_col,     // 11
      //    data_col       // 12
      //    );

      std::string kernel = jtorch::jtorch_path + "kernels/spatial_convolution_mm.cl";
      cl_context->useKernel(kernel.c_str(), "im2col_kernel");

      cl_context->setArg(0, num_kernels);
      cl_context->setArg(1, TO_TENSOR_PTR(data_im)->storage());
      cl_context->setArg(2, height);
      cl_context->setArg(3, width);
      cl_context->setArg(4, ksize_h);
      cl_context->setArg(5, ksize_w);
      cl_context->setArg(6, pad_h);
      cl_context->setArg(7, pad_w);
      cl_context->setArg(8, stride_h);
      cl_context->setArg(9, stride_w);
      cl_context->setArg(10, height_col);
      cl_context->setArg(11, width_col);
      cl_context->setArg(12, TO_TENSOR_PTR(data_col)->storage());

      uint32_t dim = 1;
      const uint32_t global_size[1] = {TO_TENSOR_PTR(data_col)->nelems()};
      cl_context->runKernel(jtorch::deviceid, dim, global_size, false);
  }

}  // namespace jtorch