#include <sstream>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include "jtorch/torch_stage.h"
#include "jtorch/linear.h"
#include "jtorch/parallel.h"
#include "jtorch/reshape.h"
#include "jtorch/sequential.h"
#include "jtorch/spatial_contrastive_normalization.h"
#include "jtorch/spatial_convolution.h"
#include "jtorch/spatial_convolution_map.h"
#include "jtorch/spatial_divisive_normalization.h"
#include "jtorch/spatial_lp_pooling.h"
#include "jtorch/spatial_max_pooling.h"
#include "jtorch/spatial_subtractive_normalization.h"
#include "jtorch/tanh.h"
#include "jtorch/threshold.h"
#include "jtorch/join_table.h"
#include "jtorch/transpose.h"
#include "jtorch/identity.h"

#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }

namespace jtorch {

  static std::mutex torch_init_lock_;
  static bool torch_init_;

  TorchStage::TorchStage() {
    output = NULL; 
  }

  TorchStage::~TorchStage() {
    
  }

  TorchStage* TorchStage::loadFromFile(const std::string& file) {
    TorchStage* ret = NULL;
    std::ifstream ifile(file.c_str(), std::ios::in|std::ios::binary);
    if (ifile.is_open()) {
      ifile.seekg(0, std::ios::beg);
      // Now recursively load the network
      std::cout << "Loading torch model..." << std::endl;
      ret = TorchStage::loadFromFile(ifile);
    } else {
      std::stringstream ss;
      ss << "HandNet::loadFromFile() - ERROR: Could not open modelfile";
      ss << " file " << file << std::endl;
      throw std::runtime_error(ss.str());
    }
    return ret;
  }

  TorchStage* TorchStage::loadFromFile(std::ifstream& ifile) { 
    // Read in the enum type:
    int type;
    ifile.read(reinterpret_cast<char*>(&type), sizeof(type));
    switch (type) {
    case SEQUENTIAL_STAGE:
      std::cout << "\tLoading Sequential..." << std::endl;
      return Sequential::loadFromFile(ifile);
    case PARALLEL_STAGE:
      std::cout << "\tLoading Parallel..." << std::endl;
      return Parallel::loadFromFile(ifile);
    case TANH_STAGE:
      std::cout << "\tLoading Tanh..." << std::endl;
      return Tanh::loadFromFile(ifile);
    case THRESHOLD_STAGE:
      std::cout << "\tLoading Threshold..." << std::endl;
      return Threshold::loadFromFile(ifile);
    case LINEAR_STAGE:
      std::cout << "\tLoading Linear..." << std::endl;
      return Linear::loadFromFile(ifile);
    case RESHAPE_STAGE:
      std::cout << "\tLoading Reshape..." << std::endl;
      return Reshape::loadFromFile(ifile);
    case SPATIAL_CONVOLUTION_STAGE:
      std::cout << "\tLoading SpatialConvolution..." << std::endl;
      return SpatialConvolution::loadFromFile(ifile);
    case SPATIAL_CONVOLUTION_MAP_STAGE:
      std::cout << "\tLoading SpatialConvolutionMap..." << std::endl;
      return SpatialConvolutionMap::loadFromFile(ifile);
    case SPATIAL_LP_POOLING_STAGE:
      std::cout << "\tLoading SpatialLPPooling..." << std::endl;
      return SpatialLPPooling::loadFromFile(ifile);
    case SPATIAL_MAX_POOLING_STAGE:
      std::cout << "\tLoading SpatialMaxPooling..." << std::endl;
      return SpatialMaxPooling::loadFromFile(ifile);
    case SPATIAL_SUBTRACTIVE_NORMALIZATION_STAGE:
      std::cout << "\tLoading SpatialSubtractiveNormalization..." << std::endl;
      return SpatialSubtractiveNormalization::loadFromFile(ifile);
    case SPATIAL_DIVISIVE_NORMALIZATION_STAGE:
      std::cout << "\tLoading SpatialDivisiveNormalization..." << std::endl;
      return SpatialDivisiveNormalization::loadFromFile(ifile);
    case SPATIAL_CONTRASTIVE_NORMALIZATION_STAGE:
      std::cout << "\tLoading SpatialContrastiveNormalization..." << std::endl;
      return SpatialContrastiveNormalization::loadFromFile(ifile);
    case JOIN_TABLE_STAGE:
      std::cout << "\tLoading JoinTable..." << std::endl;
      return JoinTable::loadFromFile(ifile);
    case TRANSPOSE_STAGE:
      std::cout << "\tLoading Transpose..." << std::endl;
      return Transpose::loadFromFile(ifile);
    case IDENTITY_STAGE:
      std::cout << "\tLoading Identity..." << std::endl;
      return Identity::loadFromFile(ifile);
    case SELECT_TABLE_STAGE:
      std::cout << "\tLoading SelectTable..." << std::endl;
      throw std::runtime_error("TODO: Implement this");
    case SPATIAL_UP_SAMPLING_NEAREST_STAGE:
      std::cout << "\tLoading SpatialUpSamplingNearestStage..." << std::endl;
      throw std::runtime_error("TODO: Implement this");
    case C_ADD_TABLE_STAGE:
      std::cout << "\tLoading CAddTable..." << std::endl;
      throw std::runtime_error("TODO: Implement this");
    default:
      throw std::runtime_error("TorchStage::loadFromFile() - ERROR: "
        "Node type not recognized!");
    }
  }

}  // namespace jtorch