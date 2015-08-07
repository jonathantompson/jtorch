dofile(jtorch_root..'/lua/save_sequential_node.lua')
dofile(jtorch_root..'/lua/save_parallel_table_node.lua')
dofile(jtorch_root..'/lua/save_tanh_node.lua')
dofile(jtorch_root..'/lua/save_threshold_node.lua')
dofile(jtorch_root..'/lua/save_linear_node.lua')
dofile(jtorch_root..'/lua/save_reshape_node.lua')
dofile(jtorch_root..'/lua/save_spatial_convolution_node.lua')
dofile(jtorch_root..'/lua/save_spatial_convolution_cuda_node.lua')
dofile(jtorch_root..'/lua/save_spatial_convolution_mm_node.lua')
dofile(jtorch_root..'/lua/save_spatial_convolution_map_node.lua')
function saveFloatTensorSafe(ofile, tensor)
  -- We need to make sure that the underlining storage is the same size as the
  -- tensor header before saving to file
  local sz = 1
  for i = 1, tensor:dim() do
    sz = sz * tensor:size(i)
  end
  assert(tensor:storage():size() == sz, 
    'underlining storage is not the same size as the tensor')
  ofile:writeFloat(tensor:storage())
end

dofile(jtorch_root..'/lua/save_spatial_lp_pooling_node.lua')
dofile(jtorch_root..'/lua/save_spatial_max_pooling_node.lua')
dofile(jtorch_root..'/lua/save_spatial_subtractive_normalization_node.lua')
dofile(jtorch_root..'/lua/save_spatial_divisive_normalization_node.lua')
dofile(jtorch_root..'/lua/save_spatial_contrastive_normalization_node.lua')
dofile(jtorch_root..'/lua/save_join_table_node.lua')
dofile(jtorch_root..'/lua/save_transpose_node.lua')
dofile(jtorch_root..'/lua/save_identity_node.lua')
dofile(jtorch_root..'/lua/save_select_table_node.lua')
dofile(jtorch_root..'/lua/save_spatial_up_sampling_nearest_node.lua')
dofile(jtorch_root..'/lua/save_c_add_table_node.lua')
dofile(jtorch_root..'/lua/save_spatial_dropout.lua')

function saveNNNode(node, ofile)
  -- Just send the node off to the correct routine depending on it's type
  -- Note that the type enum must match 
  -- jtorch/torch_stage.h
  class_str = torch.typename(node)
  print("saving " .. class_str .. "...")
  if (class_str == "nn.Sequential") then
     ofile:writeInt(1)
     saveSequentialNode(node, ofile)
  elseif (class_str == "nn.ParallelTable") then
     ofile:writeInt(2)
     saveParallelTableNode(node, ofile)
  elseif (class_str == "nn.Tanh") then
     ofile:writeInt(3)
     saveTanhNode(node, ofile)
  elseif (class_str == "nn.Threshold") then
     ofile:writeInt(4)
     saveThresholdNode(node, ofile)
  elseif (class_str == "nn.ReLU") then
     -- Note: nn.ReLU gets saved with same index as nn.Threshold
     ofile:writeInt(4)
     saveThresholdNode(node, ofile)
  elseif (class_str == 'cudnn.ReLU') then
     -- Note: cudnn.ReLU gets saved with same index as nn.Threshold
     ofile:writeInt(4)
     node.threshold = 0  -- These don't exist for cudnn.ReLU
     node.val = 0
     saveThresholdNode(node, ofile)
  elseif (class_str == "nn.Linear") then
     ofile:writeInt(5)
     saveLinearNode(node, ofile)
  elseif (class_str == "nn.Reshape") then
     ofile:writeInt(6)
     saveReshapeNode(node, ofile)
  elseif (class_str == "nn.SpatialConvolution") then
     ofile:writeInt(7)
     saveSpatialConvolutionNode(node, ofile)
  elseif (class_str == "nn.SpatialConvolutionCUDA") then
     -- Note: SpatialConvolutionCUDA gets saved with same index
     --       as SpatialConvolution
     ofile:writeInt(7)
     saveSpatialConvolutionCUDANode(node, ofile)
  elseif (class_str == "nn.SpatialConvolutionMap") then
     ofile:writeInt(8)
     saveSpatialConvolutionMapNode(node, ofile)
  elseif (class_str == "nn.SpatialLPPooling") then
     ofile:writeInt(9)
     saveSpatialLPPoolingNode(node, ofile)
  elseif (class_str == "nn.SpatialMaxPooling") then
     ofile:writeInt(10)
     saveSpatialMaxPoolingNode(node, ofile)
  elseif (class_str == "nn.SpatialMaxPoolingCUDA") then
     -- Note: SpatialMaxPoolingCUDA gets saved with the same index
     --       as SpatialMaxPooling
     ofile:writeInt(10)
     saveSpatialMaxPoolingNode(node, ofile)
  elseif ((class_str == "nn.SpatialSubtractiveNormalization") or 
          (class_str == "nn.SpatialSubtractiveNormalizationBatch")) then
     ofile:writeInt(11)
     saveSpatialSubtractiveNormalizationNode(node, ofile)
  elseif ((class_str == "nn.SpatialDivisiveNormalization") or 
          (class_str == "nn.SpatialDivisiveNormalizationBatch")) then
     ofile:writeInt(12)
     saveSpatialDivisiveNormalizationNode(node, ofile)
  elseif ((class_str == "nn.SpatialContrastiveNormalization") or 
          (class_str == "nn.SpatialContrastiveNormalizationBatch")) then
     ofile:writeInt(13)
     saveSpatialContrastiveNormalizationNode(node, ofile)
  elseif (class_str == "nn.JoinTable") then
     ofile:writeInt(14)
     saveJoinTableNode(node, ofile)
  elseif (class_str == "nn.Transpose") then
     ofile:writeInt(15)
     saveTransposeNode(node, ofile)
  elseif (class_str == "nn.Identity") then
     ofile:writeInt(16)
     saveIdentityNode(node, ofile)    
  elseif (class_str == "nn.SelectTable") then
     ofile:writeInt(17)
     saveSelectTableNode(node, ofile)
  elseif (class_str == "nn.SpatialUpSamplingNearest") then
     ofile:writeInt(18)
     saveSpatialUpSamplingNearestNode(node, ofile)
  elseif (class_str == "nn.CAddTable") then
     ofile:writeInt(19)
     saveCAddTableNode(node, ofile)
  elseif ((class_str == "nn.SpatialConvolutionMM") or 
          (class_str == "nn.SpatialConvolutionMMOut")) then
     ofile:writeInt(20)
     saveSpatialConvolutionMMNode(node, ofile)
  elseif (class_str == 'nn.SpatialDropout') then
     ofile:writeInt(21)
     saveSpatialDropoutNode(node, ofile)
  else
     error('Node type ' .. class_str .. ' is not recognized.')
     return
  end
end

