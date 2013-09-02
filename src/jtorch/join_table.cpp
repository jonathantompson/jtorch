#include "jtorch/join_table.h"
#include "jtorch/tensor.h"
#include "jtorch/table.h"
#include "jtil/exceptions/wruntime_error.h"
#include "jtil/threading/thread.h"
#include "jtil/threading/callback.h"
#include "jtil/threading/thread_pool.h"
#include "jtil/data_str/vector_managed.h"
#include "jcl/jcl.h"

#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }

using namespace jtil::threading;
using namespace jtil::math;
using namespace jtil::data_str;

namespace jtorch {

  JoinTable::JoinTable() {
    output = NULL;
  }

  JoinTable::~JoinTable() {
    SAFE_DELETE(output);
  }


  TorchStage* JoinTable::loadFromFile(std::ifstream& file) {
    int32_t output_dim;
    file.read((char*)(&output_dim), sizeof(output_dim));
    // But we don't really use it...
    return new JoinTable();
  }

  void JoinTable::init(TorchData& input) {
    if (input.type() != TorchDataType::TABLE_DATA) {
      throw std::wruntime_error("Parallel::forwardProp() - "
        "Table expected!");
    }
    Table& in = (Table&)input;

    if (in.tableSize() == 0) {
      throw std::wruntime_error("Parallel::forwardProp() - "
        "Empty input Table!");
    }

    // Check that it is a table of FloatTensors (since tables can be nested)
    int size = 0;
    for (uint32_t i = 0; i < in.tableSize(); i++) {
      if (in(i)->type() != TENSOR_DATA) {
        throw std::wruntime_error("Parallel::forwardProp() - "
          "Table of float tensors expected!");
      }
      size += in(i)->dataSize();
    }
     
    if (output != NULL && size != static_cast<int>(output->dataSize())) {
      SAFE_DELETE(output);
    }

    if (output == NULL) {
      Int3 out_dim(size, 1, 1);
      output = new Tensor<float>(out_dim);
    }
  }

  void JoinTable::forwardProp(TorchData& input) {
    init(input);

    Table& in = (Table&)input;

    // Copy each table element's raw data into the output
    std::string kernel = jtorch::jtorch_path + "kernels/join_table.cl";
    cl_context->useKernel(kernel.c_str(), "JoinTable1D");
    int out_offset = 0;
    for (uint32_t i = 0; i < in.tableSize(); i++) {
      Int3 local_worgroup_size;
      Tensor<float>* cur_input = (Tensor<float>*)in(i);
      cl_context->setArg(0, cur_input->data());
      cl_context->setArg(1, ((Tensor<float>*)output)->data());
      cl_context->setArg(2, out_offset);
      cl_context->runKernel1D(jtorch::deviceid, cur_input->dataSize(),
        false);

      out_offset += cur_input->dataSize();
    }
  }

}  // namespace jtorch
