function jtorch._saveSpatialConvolutionCUDANode(node, ofile)
  -- The layout is as follows:
  -- 1. filter width (int)
  -- 2. filter height (int)
  -- 3. filter input features (int)
  -- 4. filter output features (int)
  -- 5. padding (zero for CUDA)
  -- 6. filter weights (float array)
  -- 7. filter Biases (float)

  ofile:writeInt(node.kW)
  ofile:writeInt(node.kH)
  ofile:writeInt(node.nInputPlane)
  ofile:writeInt(node.nOutputPlane)
  ofile:writeInt(0)

  local fanin = node.nInputPlane

  -- TODO: Vectorize this! (transpose first and then save that out)
  for i=1,(node.nOutputPlane) do
    for j=1,(node.nInputPlane) do
      for v=1,node.kH do
        for u=1,node.kW do
          -- The only difference is the reordering of weights
          ofile:writeFloat(node.weight[{j, v, u, i}])
        end
      end
    end
  end
  
  assert(node.bias:dim() == 1, 'bias vector is not 1D!')
  jtorch._saveFloatTensorSafe(ofile, node.bias)
end
