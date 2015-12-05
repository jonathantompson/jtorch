#include "jtorch/torch_stage.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "jcl/opencl_context.h"
#include "jtorch/c_add_table.h"
#include "jtorch/concat.h"
#include "jtorch/concat_table.h"
#include "jtorch/identity.h"
#include "jtorch/join_table.h"
#include "jtorch/linear.h"
#include "jtorch/mul_constant.h"
#include "jtorch/narrow.h"
#include "jtorch/parallel_table.h"
#include "jtorch/reshape.h"
#include "jtorch/select_table.h"
#include "jtorch/sequential.h"
#include "jtorch/spatial_batch_normalization.h"
#include "jtorch/spatial_contrastive_normalization.h"
#include "jtorch/spatial_convolution_map.h"
#include "jtorch/spatial_convolution_mm.h"
#include "jtorch/spatial_convolution.h"
#include "jtorch/spatial_divisive_normalization.h"
#include "jtorch/spatial_dropout.h"
#include "jtorch/spatial_lp_pooling.h"
#include "jtorch/spatial_max_pooling.h"
#include "jtorch/spatial_subtractive_normalization.h"
#include "jtorch/spatial_up_sampling_nearest.h"
#include "jtorch/tanh.h"
#include "jtorch/threshold.h"
#include "jtorch/transpose.h"

namespace jtorch {

TorchStage::TorchStage() { output = nullptr; }

TorchStage::~TorchStage() {}

std::unique_ptr<TorchStage> TorchStage::loadFromFile(const std::string& file) {
  std::unique_ptr<TorchStage> ret;
  std::ifstream ifile(file.c_str(), std::ios::in | std::ios::binary);
  if (ifile.is_open()) {
    ifile.seekg(0, std::ios::beg);
    // Now recursively load the network
    ret = TorchStage::loadFromFile(ifile);
    ifile.close();
  } else {
    std::cout << "TorchStage::loadFromFile() - ERROR: Could not open modelfile";
    std::cout << " file " << file << std::endl;
    RASSERT(false);
  }
  return ret;
}

std::unique_ptr<TorchStage> TorchStage::loadFromFile(std::ifstream& ifile) {
  // Read in the enum type:

  int type;
  ifile.read(reinterpret_cast<char*>(&type), sizeof(type));

  // Now load in the module
  std::unique_ptr<TorchStage> node;
  switch (type) {
    case SEQUENTIAL_STAGE:
      node = Sequential::loadFromFile(ifile);
      break;
    case PARALLEL_TABLE_STAGE:
      node = ParallelTable::loadFromFile(ifile);
      break;
    case TANH_STAGE:
      node = Tanh::loadFromFile(ifile);
      break;
    case THRESHOLD_STAGE:
      node = Threshold::loadFromFile(ifile);
      break;
    case LINEAR_STAGE:
      node = Linear::loadFromFile(ifile);
      break;
    case RESHAPE_STAGE:
      node = Reshape::loadFromFile(ifile);
      break;
    case SPATIAL_CONVOLUTION_STAGE:
      node = SpatialConvolution::loadFromFile(ifile);
      break;
    case SPATIAL_CONVOLUTION_MAP_STAGE:
      node = SpatialConvolutionMap::loadFromFile(ifile);
      break;
    case SPATIAL_LP_POOLING_STAGE:
      node = SpatialLPPooling::loadFromFile(ifile);
      break;
    case SPATIAL_MAX_POOLING_STAGE:
      node = SpatialMaxPooling::loadFromFile(ifile);
      break;
    case SPATIAL_SUBTRACTIVE_NORMALIZATION_STAGE:
      node = SpatialSubtractiveNormalization::loadFromFile(ifile);
      break;
    case SPATIAL_DIVISIVE_NORMALIZATION_STAGE:
      node = SpatialDivisiveNormalization::loadFromFile(ifile);
      break;
    case SPATIAL_CONTRASTIVE_NORMALIZATION_STAGE:
      node = SpatialContrastiveNormalization::loadFromFile(ifile);
      break;
    case JOIN_TABLE_STAGE:
      node = JoinTable::loadFromFile(ifile);
      break;
    case TRANSPOSE_STAGE:
      node = Transpose::loadFromFile(ifile);
      break;
    case IDENTITY_STAGE:
      node = Identity::loadFromFile(ifile);
      break;
    case SELECT_TABLE_STAGE:
      node = SelectTable::loadFromFile(ifile);
      break;
    case SPATIAL_UP_SAMPLING_NEAREST_STAGE:
      node = SpatialUpSamplingNearest::loadFromFile(ifile);
      break;
    case C_ADD_TABLE_STAGE:
      node = CAddTable::loadFromFile(ifile);
      break;
    case SPATIAL_CONVOLUTION_MM_STAGE:
      node = SpatialConvolutionMM::loadFromFile(ifile);
      break;
    case SPATIAL_DROPOUT_STAGE:
      node = SpatialDropout::loadFromFile(ifile);
      break;
    case SPATIAL_BATCH_NORMALIZATION_STAGE:
      node = SpatialBatchNormalization::loadFromFile(ifile);
      break;
    case CONCAT_STAGE:
      node = Concat::loadFromFile(ifile);
      break;
    case NARROW_STAGE:
      node = Narrow::loadFromFile(ifile);
      break;
    case MUL_CONSTANT_STAGE:
      node = MulConstant::loadFromFile(ifile);
      break;
    case CONCAT_TABLE_STAGE:
      node = ConcatTable::loadFromFile(ifile);
      break;
    default:
      std::cout << "TorchStage::loadFromFile() - ERROR: "
                   "Node type not recognized!"
                << std::endl;
      RASSERT(false);
  }

#if defined(DEBUG) || defined(_DEBUG)
  std::cout << "\tLoaded " << node->name() << std::endl;
#endif
  return node;
}

}  // namespace jtorch
