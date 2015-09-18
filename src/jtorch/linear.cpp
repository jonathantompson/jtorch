#include "jtorch/linear.h"
#include "jtorch/tensor.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"
#include "jcl/jcl.h"

using namespace jcl::threading;
using namespace jcl::math;

namespace jtorch {

static const char* kLinearKernel =
"    __kernel void MatVecMultSimple(\n"
"      /* Y = A * X (matrix-vector mulitply)*/\n"
"      __global const float* A,  /* 0  --> Size M (rows) x N (cols) stored column major */\n"
"      __global const float* X,  /* 1  --> Size N */\n"
"      __global  float* Y,       /* 2  --> Size M */\n"
"      const int M,              /* 3 */\n"
"      const int N) {            /* 4 */\n"
"      const int i = get_global_id(0);  /* row index */\n"
"      float sum = 0;\n"
"      /* Perform the linear accumulation */\n"
"      for (int k = 0; k < N; k++) {\n"
"        sum += A[i + M * k] * X[k];\n"
"      }\n"
"      Y[i] = sum;\n"
"    }\n"
"\n"
"    #define ROW_DIM 0\n"
"    #define COL_DIM 1\n"
"\n"
"    __kernel void MatVecMultThreads(\n"
"      /* Y = A * X (matrix-vector mulitply) */\n"
"      __global const float* A,  /* 0  --> Size M (rows) x N (cols) stored column major */\n"
"      __global const float* X,  /* 1  --> Size N */\n"
"      __global  float* Y,       /* 2  --> Size M */\n"
"      __local float* work,      /* 3  --> Size M by p */\n"
"      const int M,              /* 4 */\n"
"      const int N) {            /* 5 */\n"
"      /* Compute partial dot product */\n"
"      float sum = 0;\n"
"      for (int k = get_global_id(COL_DIM); k < N; k += get_global_size(COL_DIM)) {\n"
"        sum += A[get_global_id(ROW_DIM) + M * k] * X[k];\n"
"      }\n"
"      /* Each thread stores its partial sum in WORK */\n"
"      int rows = get_local_size(ROW_DIM); /* rows in group */\n"
"      int cols = get_local_size(COL_DIM); /* initial cols in group */\n"
"      int ii = get_local_id(ROW_DIM); /* local row index in group, 0<=ii<rows */\n"
"      int jj = get_local_id(COL_DIM); /* block index in column, 0<=jj<cols */\n"
"      work[ii+rows*jj] = sum;\n"
"      barrier(CLK_LOCAL_MEM_FENCE); /* sync group */\n"
"      /* Reduce sums in log2(cols) steps */\n"
"      while (cols > 1) {\n"
"        cols >>= 1;\n"
"      if (jj < cols) { \n"
"        work[ii + rows * jj] += work[ii + rows * (jj + cols)];\n"
"      }\n"
"      barrier(CLK_LOCAL_MEM_FENCE); /* sync group */\n"
"      }\n"
"      /* Write final result in Y */\n"
"      if ( jj == 0 ) {\n"
"        Y[ get_global_id(ROW_DIM) ] = work[ii];\n"
"      }\n"
"    }\n"
"\n"
"    __kernel void Accum (\n"
"      __global  float* output,          /* 0 */\n"
"      const __global float* biases) {   /* 1 */\n"
"      const int x_out = get_global_id(0);\n"
"      output[x_out] += biases[x_out];\n"
"    }";

Linear::Linear(const uint32_t n_inputs, const uint32_t n_outputs)
    : TorchStage() {
  n_inputs_ = n_inputs;
  n_outputs_ = n_outputs;

  output.reset(new Tensor<float>(1, &n_outputs_));

  // NOTE: For efficiency we store the weight matrix transposed!
  // (we want the matrix vector multiply to be strided properly)
  uint32_t size_[2] = {n_outputs_, n_inputs_};
  weights_.reset(new Tensor<float>(2, size_));
  biases_.reset(new Tensor<float>(1, &n_outputs_));
}

Linear::~Linear() {}

void Linear::setWeights(const float* weights) { weights_->setData(weights); }

void Linear::setBiases(const float* biases) { biases_->setData(biases); }

void Linear::init(std::shared_ptr<TorchData> input) {
  // FloatTensor expected
  RASSERT(input->type() == TorchDataType::TENSOR_DATA);
  Tensor<float>* in = TO_TENSOR_PTR(input.get());
  static_cast<void>(in);
  // Check input size
  RASSERT(in->dim() == 1 && in->size()[0] == n_inputs_);
}

void Linear::forwardProp(std::shared_ptr<TorchData> input) {
  init(input);
#ifdef SIMPLE_LINEAR
  cl_context->useKernelCStr(kLinearKernel, "MatVecMultSimple");
  cl_context->setArg(0, weights_->storage());
  cl_context->setArg(1, TO_TENSOR_PTR(input.get())->storage());
  cl_context->setArg(2, TO_TENSOR_PTR(output.get())->storage());
  cl_context->setArg(3, (int)n_outputs_);
  cl_context->setArg(4, (int)n_inputs_);
  uint32_t dim = 1;
  cl_context->runKernel(jtorch::deviceid, dim, &n_outputs_, false);
#else
  cl_context->useKernelCStr(kLinearKernel, "MatVecMultThreads");

  uint32_t max_worksize =
      cl_context->queryMaxWorkgroupSizeForCurKernel(jtorch::deviceid);
  // http://www.bealto.com/gpu-gemv_v2.html
  // Try and find a good local workgroup size allocation (that is legal)
  // TODO: This is a mess.  Clean it up.
  uint32_t max_item_size[3];
  for (uint32_t i = 0; i < 3; i++) {
    max_item_size[i] = cl_context->getMaxWorkitemSize(jtorch::deviceid, i);
  }

  uint32_t p =
      std::min<int32_t>(16, std::min<int32_t>(max_item_size[1], max_worksize));
  uint32_t global_size[2] = {n_outputs_, p};
  uint32_t local_size[2] = {
      std::min<int>(n_outputs_ / p + 1,
                    cl_context->getMaxWorkgroupSize(jtorch::deviceid) / p),
      p};  // Maximum
  while ((n_outputs_ % local_size[0] != 0 ||
          local_size[0] * local_size[1] > max_worksize) &&
         local_size[0] > 1) {
    local_size[0]--;
  }

  cl_context->setArg(0, weights_->storage());
  cl_context->setArg(1, TO_TENSOR_PTR(input.get())->storage());
  cl_context->setArg(2, TO_TENSOR_PTR(output.get())->storage());
  float dummy;
  static_cast<void>(dummy);
  // setArg with nullptr --> Local memory allocation (per local workgroup)
  cl_context->setArg(3, sizeof(dummy) * local_size[0] * local_size[1], nullptr);
  cl_context->setArg(4, (int)n_outputs_);
  cl_context->setArg(5, (int)n_inputs_);
  uint32_t dim = 2;
  cl_context->runKernel(jtorch::deviceid, dim, global_size, local_size, false);
#endif

  // Now add in the bias
  cl_context->useKernelCStr(kLinearKernel, "Accum");
  cl_context->setArg(0, TO_TENSOR_PTR(output.get())->storage());
  cl_context->setArg(1, biases_->storage());
  dim = 1;
  cl_context->runKernel(jtorch::deviceid, dim, &n_outputs_, false);
}

std::unique_ptr<TorchStage> Linear::loadFromFile(std::ifstream& file) {
  int32_t n_outputs;
  int32_t n_inputs;
  file.read((char*)(&n_outputs), sizeof(n_outputs));
  file.read((char*)(&n_inputs), sizeof(n_inputs));
  std::unique_ptr<Linear> ret(new Linear(n_inputs, n_outputs));

  int32_t n_weights = n_outputs * n_inputs;
  std::unique_ptr<float[]> weights_cpu(new float[n_weights]);
  file.read((char*)(weights_cpu.get()), sizeof(weights_cpu[0]) * n_weights);

  ret->setWeights(weights_cpu.get());

  std::unique_ptr<float[]> bias_cpu(new float[n_outputs]);
  file.read((char*)(bias_cpu.get()), sizeof(bias_cpu[0]) * n_outputs);
  ret->setBiases(bias_cpu.get());

  return std::unique_ptr<TorchStage>(std::move(ret));
}

}  // namespace jtorch
