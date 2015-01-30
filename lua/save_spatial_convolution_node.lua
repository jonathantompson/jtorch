function saveSpatialConvolutionNode(node, ofile)
  -- The layout is as follows:
  -- 1. filter width (int)
  -- 2. filter height (int)
  -- 3. filter input features (int)
  -- 4. filter output features (int)
  -- 5. padding (zero for SpatialConvolution)
  -- 6. filter weights (float array)
  -- 7. filter Biases (float)

  ofile:writeInt(node.kW)
  ofile:writeInt(node.kH)
  ofile:writeInt(node.nInputPlane)
  ofile:writeInt(node.nOutputPlane)
  ofile:writeInt(0)

  local fanin = node.nInputPlane

  assert(node.weight:dim() == 4, 'weight tensor is not 4D!')
  saveFloatTensorSafe(ofile, node.weight)
  assert(node.bias:dim() == 1, 'bias vector is not 1D!')
  saveFloatTensorSafe(ofile, node.bias)

end
