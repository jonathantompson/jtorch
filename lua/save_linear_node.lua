function saveLinearNode(node, ofile)
  local noutputs = node.weight:size()[1]
  local ninputs = node.weight:size()[2]
  ofile:writeInt(noutputs)
  ofile:writeInt(ninputs)
 
  assert(node.weight:dim() == 2, 'weight matrix is not 2D!')
  saveFloatTensorSafe(ofile, node.weight)
  assert(node.bias:dim() == 1, 'bias vector is not 1D!')
  saveFloatTensorSafe(ofile, node.bias)
end
