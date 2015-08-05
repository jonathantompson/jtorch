#include "jtorch/join_table.h"
#include "jtorch/tensor.h"
#include "jtorch/table.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"
#include "jcl/data_str/vector_managed.h"
#include "jcl/jcl.h"

#define SAFE_DELETE(x) if (x != nullptr) { delete x; x = nullptr; }
#define SAFE_DELETE_ARR(x) if (x != nullptr) { delete[] x; x = nullptr; }

using namespace jcl::threading;
using namespace jcl::math;
using namespace jcl::data_str;

namespace jtorch {

  JoinTable::JoinTable(const uint32_t dimension) {
    dimension_ = dimension;
    output = nullptr;
  }

  JoinTable::~JoinTable() {
    SAFE_DELETE(output);
  }


  TorchStage* JoinTable::loadFromFile(std::ifstream& file) {
    int32_t dimension;
    file.read((char*)(&dimension), sizeof(dimension));
    dimension = dimension - 1;  // We index from 0 in C++
    // But we don't really use it...
    return new JoinTable(dimension);
  }

  void JoinTable::init(TorchData& input) {
    if (input.type() != TorchDataType::TABLE_DATA) {
      throw std::runtime_error("JoinTable::forwardProp() - "
        "Table expected!");
    }
    Table& in = (Table&)input;

    if (in.tableSize() == 0) {
      throw std::runtime_error("JoinTable::forwardProp() - "
        "Empty input Table!");
    }

    // Check that it is a table of FloatTensors
    for (uint32_t i = 0; i < in.tableSize(); i++) {
      if (in(i)->type() != TENSOR_DATA) {
        throw std::runtime_error("JoinTable::forwardProp() - "
          "Table of float tensors expected!");
      }
    }

    uint32_t dim = TO_TENSOR_PTR(in(0))->dim();
    if (dim <= dimension_) {
      throw std::runtime_error("JoinTable::forwardProp() - "
        "Input is smaller than join dimension!");
    }
    uint32_t jdim = dim - dimension_ - 1;  // dimension_=0 is the top dim

    // Make sure the dimensions OTHER than the join dimension are all the same
    for (uint32_t d = 0; d < dim; d++) {
      if (d != jdim) {
        for (uint32_t j = 1; j < in.tableSize(); j++) {
          if (TO_TENSOR_PTR(in(j))->size()[d] != TO_TENSOR_PTR(in(0))->size()[d]) {
            throw std::runtime_error("JoinTable::forwardProp() - "
              "Size mismatch!");
          }
        }
        if (output != nullptr && TO_TENSOR_PTR(output)->size()[d] != 
          TO_TENSOR_PTR(in(0))->size()[d]) {
            SAFE_DELETE(output);
        }
      }
    }

    uint32_t nelems_jdim = 0;
    for (uint32_t j = 1; j < in.tableSize(); j++) {
      nelems_jdim += TO_TENSOR_PTR(in(j))->size()[jdim];
    }

    if (output != nullptr &&
      TO_TENSOR_PTR(output)->size()[jdim] != nelems_jdim) {
      SAFE_DELETE(output);
    }

    if (output == nullptr) {
      uint32_t* size = new uint32_t[dim];
      memcpy(size, TO_TENSOR_PTR(in(0))->size(), sizeof(size[0]) * dim);
      size[dimension_] = nelems_jdim;
      output = new Tensor<float>(dim, size);
      SAFE_DELETE_ARR(size);
    }
  }

  void JoinTable::forwardProp(TorchData& input) {
    init(input);

    Table& in = (Table&)input;

    // AT THE MOMENT ONLY JOINS ALONG THE TOP DIMENSION ARE SUPPORTED
    if (dimension_ != 0) {
      throw std::runtime_error("JoinTable::forwardProp() - "
        "Only dimension=0 is supported for now");
    }

    // Copy each table element's raw data into the output
    std::string kernel = jtorch::jtorch_path + "kernels/join_table.cl";
    cl_context->useKernel(kernel.c_str(), "JoinTable1D");
    int out_offset = 0;
    for (uint32_t i = 0; i < in.tableSize(); i++) {
      Tensor<float>* cur_input = (Tensor<float>*)in(i);
      cl_context->setArg(0, cur_input->storage());
      cl_context->setArg(1, TO_TENSOR_PTR(output)->storage());
      cl_context->setArg(2, out_offset);
      uint32_t dim = 1;
      uint32_t nelem = cur_input->nelems();
      cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);

      out_offset += nelem;
    }
  }

}  // namespace jtorch
